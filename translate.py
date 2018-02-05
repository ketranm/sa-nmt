import argparse
import torch
import modelx as models
import infer
import string

# build args parser
parser = argparse.ArgumentParser(description='Training NMT')

parser.add_argument('-checkpoint', required=True,
                    help='saved checkpoit.')

parser.add_argument('-input', required=True,
                    help='Text file to translate.')
parser.add_argument('-output', default='trans.bpe', help='output file')
parser.add_argument('-beam_size', default=5, type=int,
                    help="Beam size.")
parser.add_argument('-gpuid', default=[], nargs='+', type=int,
                    help="Use CUDA")


opt = parser.parse_args()
use_gpu = len(opt.gpuid) > 0

if torch.cuda.is_available() and not use_gpu:
    print("so you should probably run with -gpus 0")

checkpoint = torch.load(opt.checkpoint)
train_opt = checkpoint['opt']
print('| train configuration')
train_opt.min_thres = -5.0
train_opt.max_thres = 7.0
#if train_opt.hard is None:
#train_opt.hard = False
print(train_opt)

model = models.make_base_model(train_opt, use_gpu, checkpoint)
if train_opt.encoder_type == "sabrnn":
    punct_idx = set()
    for p in set(string.punctuation):
        if p in model.dicts[0]:
            punct_idx.add(model.dicts[0][p])
    model.encoder.punct(punct_idx)
# over-write some options
train_opt.beam_size = opt.beam_size
agent = infer.Beam(train_opt, model)
agent.translate(opt.input, opt.output)
