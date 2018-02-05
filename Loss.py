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
from torch.autograd import Variable


def nmt_criterion(vocab_size, gpuid, pad_id=0):
    """
    Construct the standard NMT Criterion
    """
    weight = torch.ones(vocab_size)
    weight[pad_id] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if gpuid:
        crit.cuda()
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
                          .sum()
        return Statistics(loss[0], non_padding.int().sum(), num_correct)


def filter_gen_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    state:
        A dictionary which corresponds to the output of
        LossCompute.make_loss_batch(). In other words, its keys are
        {'out', 'target', 'align', 'coverage', 'attn'}. The values
        for those keys are Tensor-like or None.
    shard_size:
        The maximum size of the shards yielded by the model
    eval:
        If True, only yield the state, nothing else. Otherwise, yield shards.
    yields:
        Each yielded shard is a dict.
    side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_gen_state(state))

        # Now, the iteration:
        # split_state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
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
        scores_data = scores.data.clone()
        target = target.data.clone()

        # Coverage loss term.
        ppl = loss.data.clone()

        stats = Statistics.score(ppl, scores_data, target, 0)
        return loss, stats
