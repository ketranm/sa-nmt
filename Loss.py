"""
This file handles the details of the loss function during training.

This includes: loss criterion, training statistics, and memory optimizations.
"""
from __future__ import division
import time
import sys
import math
import torch
import torch.nn as nn


def nmt_criterion(vocab_size, pad_id=0):
    """
    Construct the standard NMT Criterion
    """
    weight = torch.ones(vocab_size)
    weight[pad_id] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    return crit


class Statistics:
    """
    Training loss function statistics.
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        print(self.loss, self.n_words)
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, uidx, max_updates, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, uidx,  max_updates,
               self.accuracy(),
               self.ppl(),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, optim):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", optim.lr)

    @staticmethod
    def score(loss, scores, targ, pad):
        pred = scores.max(1)[1]
        non_padding = targ.ne(pad)
        num_correct = pred.eq(targ) \
                          .masked_select(non_padding).int() \
                          .sum().item()
        return Statistics(loss, non_padding.int().sum().item(), num_correct)


def filter_gen_state(state):
    for k, v in state.items():
        if v is not None:
            yield k, v


def new_split(x, size):
    xs = []
    for u in torch.split(x, size):
        v = u.detach()
        if u.requires_grad:
            v.requires_grad_(True)
        xs += [v]
    return tuple(xs)


def shards(state, shard_size, eval=False):
    if eval:
        yield state
    else:
        non_none = dict(filter_gen_state(state))

        keys, values = zip(*((k, new_split(v, shard_size))
                             for k, v in non_none.items()))
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for i, k in enumerate(keys):
            dv = [v.grad for v in values[i] if v.grad is not None]
            if dv:
                dv = torch.cat(dv)
                variables += [(state[k], dv)]

        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)


class LossCompute:
    def __init__(self, generator, crit):
        self.generator = generator
        self.crit = crit

    def make_loss_batch(self, outputs, targets):
        """
        Create all the variables that need to be sharded.
        This needs to match compute loss exactly.
        """
        return {"out": outputs,
                "target": targets}

    def compute_loss(self, out, target):

        def bottle(v):
            return v.view(-1, v.size(2))

        target = target.view(-1)

        # Standard generator.
        scores = self.generator(bottle(out))
        loss = self.crit(scores, target)
        scores_data = scores.detach()
        target = target.clone()

        # Coverage loss term.
        stats = Statistics.score(loss.item(), scores_data, target, 0)
        return loss, stats
