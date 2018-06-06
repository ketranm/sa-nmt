import argparse
import torch
from Iterator import TextIterator
import models
from itertools import zip_longest
import random
import Loss
import opts
import os
import math
import subprocess
from infer import Beam
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description='train.py')

# Data and loading options
parser.add_argument('-datasets', required=True, default=[],
                    nargs='+', type=str,
                    help='source_file target_file.')
parser.add_argument('-valid_datasets', required=True, default=[],
                    nargs='+', type=str,
                    help='valid_source valid target files.')
parser.add_argument('-beam_size', default=12, type=int, help="beam size")
# dictionaries
parser.add_argument('-dicts', required=True, default=[],
                    nargs='+',
                    help='source_vocab.pkl target_vocab.pkl files.')

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.preprocess_opts(parser)

opt = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for reproducibility
torch.manual_seed(opt.seed)
random.seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)

print(opt)


# batch preparation
def prepare_data(seqs_x, seqs_y):
    mb = [(seqs_x[i], seqs_y[i]) for i in range(len(seqs_x))]
    mb.sort(key=lambda x: len(x[0]), reverse=True)
    xs = torch.LongTensor(
        list(zip_longest(*map(lambda x: x[0], mb), fillvalue=0))).to(device)
    ys = torch.LongTensor(
        list(zip_longest(*map(lambda x: x[1], mb), fillvalue=0))).to(device)
    lengths_x = [len(x[0]) for x in mb]
    return xs, ys, lengths_x


def eval(model, criterion, valid_data):
    stats = Loss.Statistics()
    model.eval()
    loss = Loss.LossCompute(model.generator, criterion)
    for src, tgt in valid_data:
        src, tgt, src_lengths = prepare_data(src, tgt, True)
        outputs = model(src, tgt[:-1], src_lengths)
        gen_state = loss.make_loss_batch(outputs, tgt[1:])
        _, batch_stats = loss.compute_loss(**gen_state)
        stats.update(batch_stats)
    model.train()
    return stats


def init_uniform(model, init_range=0.04):
    """Simple uniform initialization of all the weights"""
    for p in model.parameters():
        p.data.uniform_(-init_range, init_range)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def check_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def train(opt):
    print('| build data iterators')
    train = TextIterator(*opt.datasets, *opt.dicts,
                         src_vocab_size=opt.src_vocab_size,
                         tgt_vocab_size=opt.tgt_vocab_size,
                         batch_size=opt.batch_size,
                         max_seq_length=opt.max_seq_length)

    valid = TextIterator(*opt.valid_datasets, *opt.dicts,
                         src_vocab_size=opt.src_vocab_size,
                         tgt_vocab_size=opt.tgt_vocab_size,
                         batch_size=opt.batch_size,
                         max_seq_length=opt.max_seq_length)

    if opt.src_vocab_size < 0:
        opt.src_vocab_size = len(train.source_dict)
    if opt.tgt_vocab_size < 0:
        opt.tgt_vocab_size = len(train.target_dict)

    print('| vocabulary size. source = %d; target = %d' %
          (opt.src_vocab_size, opt.tgt_vocab_size))
    dicts = [train.source_dict, train.target_dict]

    crit = Loss.nmt_criterion(opt.tgt_vocab_size, 0).to(device)
    if opt.train_from != '':
        print('| Load trained model!')
        checkpoint = torch.load(opt.train_from)
        model = models.make_base_model(opt, checkpoint)
    else:
        model = models.make_base_model(opt)
        init_uniform(model)
    model.to(device)
    if opt.encoder_type in ["sabrnn", "fabrnn"]:
        print('Add punctuation constrain!')
        model.encoder.punct(train.src_punct)
    print(model)
    model.dicts = dicts
    check_model_path()
    tally_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  factor=opt.learning_rate_decay,
                                  patience=0)
    uidx = 0  # number of updates
    estop = False
    min_lr = opt.learning_rate * math.pow(opt.learning_rate_decay, 5)
    best_bleu = -1
    for eidx in range(1, opt.epochs + 1):
        closs = Loss.LossCompute(model.generator, crit)
        tot_loss = 0
        total_stats = Loss.Statistics()
        report_stats = Loss.Statistics()
        for x, y in train:
            model.zero_grad()
            src, tgt, lengths_x = prepare_data(x, y)
            out = model(src, tgt[:-1], lengths_x)
            gen_state = closs.make_loss_batch(out, tgt[1:])
            shard_size = opt.max_generator_batches
            batch_size = len(lengths_x)
            batch_stats = Loss.Statistics()
            for shard in Loss.shards(gen_state, shard_size):
                loss, stats = closs.compute_loss(**shard)
                loss.div(batch_size).backward()
                batch_stats.update(stats)
                tot_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           opt.max_grad_norm)
            optimizer.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)
            uidx += 1
            if uidx % opt.report_every == 0:
                report_stats.output(eidx, uidx, opt.max_updates,
                                    total_stats.start_time)
                report_stats = Loss.Statistics()

            if uidx % opt.eval_every == 0:
                valid_stats = eval(model, crit, valid)
                # maybe adjust learning rate
                scheduler.step(valid_stats.ppl())
                cur_lr = optimizer.param_groups[0]['lr']
                print('Validation perplexity %d: %g' %
                      (uidx, valid_stats.ppl()))
                print('Learning rate: %g' % cur_lr)
                if cur_lr < min_lr:
                    print('Reaching minimum learning rate. Stop training!')
                    estop = True
                    break
                model_state_dict = model.state_dict()
                if eidx >= opt.start_checkpoint_at:
                    checkpoint = {
                        'model': model_state_dict,
                        'opt': opt,
                        'dicts': dicts
                    }

                    # evaluate with BLEU score
                    inference = Beam(opt, model)
                    output_bpe = opt.save_model + '.bpe'
                    output_txt = opt.save_model + '.txt'
                    inference.translate(opt.valid_datasets[0], output_bpe)
                    model.train()
                    subprocess.call("sed 's/@@ //g' {:s} > {:s}"
                                    .format(output_bpe, output_txt),
                                    shell=True)
                    ref = opt.valid_datasets[1][:-4]
                    subprocess.call("sed 's/@@ //g' {:s} > {:s}"
                                    .format(opt.valid_datasets[1], ref),
                                    shell=True)
                    cmd = "perl data/multi-bleu.perl {} < {}" \
                        .format(ref, output_txt)
                    p = subprocess.Popen(cmd,
                                         shell=True,
                                         stdout=subprocess.PIPE) \
                        .stdout.read().decode('utf-8')
                    bleu = re.search("[\d]+.[\d]+", p)
                    bleu = float(bleu.group())
                    print('Validation BLEU %d: %g' % (uidx, bleu))
                    if bleu > best_bleu:
                        best_bleu = bleu
                        torch.save(checkpoint, '%s_best.pt' % opt.save_model)
                        print('Saved model: %d | BLEU %.2f' % (uidx, bleu))

            if uidx >= opt.max_updates:
                print('Finishing after {:d} iterations!'.format(uidx))
                estop = True
                break
        if estop:
            break


train(opt)
