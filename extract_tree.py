import argparse
import torch
from torch.autograd import Variable
import modelx as models
import networkx as nx
from networkx.algorithms.tree import maximum_spanning_arborescence
import string

# build args parser
parser = argparse.ArgumentParser(description='Training NMT')

parser.add_argument('-checkpoint', required=True,
                    help='saved checkpoit.')

parser.add_argument('-input', required=True,
                    help='Text file to translate.')
parser.add_argument('-output', default='tree.txt', help='output file')
parser.add_argument('-gpuid', default=[], nargs='+', type=int,
                    help="Use CUDA")


opt = parser.parse_args()
use_gpu = True
if torch.cuda.is_available() and not use_gpu:
    print("so you should probably run with -gpus 0")

checkpoint = torch.load(opt.checkpoint)
train_opt = checkpoint['opt']
print('| train configuration')
print(train_opt)
use_gpu = len(train_opt.gpuid) > 0
model = models.make_base_model(train_opt, use_gpu, checkpoint)
if train_opt.encoder_type in ["sabrnn", "fabrnn"]:
    punct_idx = set()
    for p in set(string.punctuation):
        if p in model.dicts[0]:
            punct_idx.add(model.dicts[0][p])
    print('Add punctuation constraint')
    model.encoder.punct(punct_idx)
# get the encoder
encoder = model.encoder
dicts = model.dicts
tt = torch.cuda if use_gpu else torch


def encode_string(ss):
    ss = ss.split()
    ss = [dicts[0].get(w, 1) for w in ss]
    ss = Variable(tt.LongTensor(ss).view(-1, 1),
                  volatile=True)
    return ss


def collapse_bpe(s, score):
    """Collapse BPEed tokens
    Args:
        s: a bped sentence
        score: beta in the paper, a 2D tensor of p(z|s),
        sum over the last dimension should be 1
    """
    punct = set(string.punctuation)
    # (1) identify bpe
    tokens = s.split()
    bpe = []
    indices = []
    punct_idx = []

    for i, w in enumerate(tokens):
        if w in punct:
            punct_idx += [i]
        if w.endswith("@@"):
            bpe += [i]
        else:
            if len(bpe) == 0:
                indices += [[i]]
            else:
                bpe += [i]  # add the last bped token
                indices += [bpe]
                bpe = []
    # collapsing from here
    s_ = []
    for bpe in indices:
        # (1) collapse heads
        s_.append(score[:, bpe].sum(1).view(-1, 1))
    s_ = torch.cat(s_, 1)
    collapsed_score = []
    for bpe in indices:
        # (2) collapse childs
        collapsed_score += [s_[bpe, :].sum(0).view(1, -1)]
    collapsed_score = torch.cat(collapsed_score, 0)
    s = s.replace("@@ ", "")  # the original string
    return s, collapsed_score


def build_graph(score):
    """Build graph from potential score matrix
    Args:
        score: FloatTensor (n, n), score.sum(1) = 1
    Returns:
        a graph object
    """
    # return a list of (parent, child, weight)
    arcs = []
    n = score.size(0)
    # find the root first
    val, idx = score.diag().max(0)
    arcs.append((0, idx[0] + 1, val[0]))
    for i in range(n):
        for j in range(n):
            if i == j:  # root on the diagonal
                continue
                # arcs.append([0, i+1, score[i, j]])
            else:
                arcs.append([j+1, i+1, score[i, j]])
    g = nx.DiGraph()
    g.add_weighted_edges_from(arcs)
    return g


def mst(score):
    """Get spaning tree from the adjacent matrix"""
    g = build_graph(score)
    mst = maximum_spanning_arborescence(g)
    tree = []
    for e in mst.edges():
        head, child = e
        tree.append('%s->%s' % (head, child))
    return ' '.join(tree)


def renorm(m):
    x = m.exp()
    x = x / x.sum(1, keepdim=True)
    return x.log()


def collapse():
    fw = open(opt.output, 'w')
    for line in open(opt.input):
        x = encode_string(line)
        model.encoder(x)
        if train_opt.encoder_type == 'sabrnn':
            score = model.encoder.tree_attn.edge_score.squeeze(0)
        else:
            score = model.encoder.attn.score.squeeze(0).log()
        s, score = collapse_bpe(line, score.data)
        try:
            tree = mst(score)
            out = '%s ||| %s\n' % (s.strip(), tree)
            fw.write(out)
        except:
            pass
    fw.close()


collapse()
