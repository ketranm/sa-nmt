import numpy
import random
import pickle as pkl
import gzip
from tempfile import mkstemp
import os
import string


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source_file, target_file,
                 source_dict, target_dict,
                 batch_size=32,
                 batch_tokens=0,
                 max_seq_length=50,
                 src_vocab_size=-1,
                 tgt_vocab_size=-1):

        self.source_file = source_file
        self.target_file = target_file
        self.shuffle([source_file, target_file])
        self.source = fopen(source_file + '.shuf', 'r')
        self.target = fopen(target_file + '.shuf', 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)
        self.src_punct = set()
        for p in set(string.punctuation):
            if p in self.source_dict:
                self.src_punct.add(self.source_dict[p])

        self.batch_tokens = batch_tokens
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        if src_vocab_size < 0:
            self.src_vocab_size = len(self.source_dict)
        if tgt_vocab_size < 0:
            self.tgt_vocab_size = len(self.target_dict)

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * 20

        self.end_of_data = False

    def __iter__(self):
        return self

    @staticmethod
    def shuffle(files):
        print('shuffle %s | %s' % (files[0], files[1]))
        tf_os, tpath = mkstemp()
        tf = open(tpath, 'w')
        fds = [open(ff) for ff in files]
        punct = set(string.punctuation)

        def punct_count(ws):
            p = [w for w in ws if w in punct]
            return len(p)

        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            tokens = [l.split() for l in lines]
            lengths = [len(t) for t in tokens]
            if min(lengths) * 2 < max(lengths):
                continue
            if min(lengths) < 1:
                continue
            # remove lines that have many punctuations
            n_punct = punct_count(tokens[0])
            if n_punct * 4 > lengths[0]:
                continue

            print("|||".join(lines), file=tf)

        [ff.close() for ff in fds]
        tf.close()

        tf = open(tpath, 'r')
        lines = tf.readlines()
        random.shuffle(lines)

        fds = [open(ff + '.shuf', 'w') for ff in files]

        for l in lines:
            s = l.strip().split('|||')
            for ii, fd in enumerate(fds):
                print(s[ii], file=fd)

        [ff.close() for ff in fds]

        os.remove(tpath)
        return

    def reset(self):
        self.shuffle([self.source_file, self.target_file])
        self.source.seek(0)
        self.target.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        batch_tokens = 0

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer)

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break

                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            tlen = numpy.array([len(t) for t in self.target_buffer])
            tidx = tlen.argsort()

            _sbuf = [self.source_buffer[i] for i in tidx]
            _tbuf = [self.target_buffer[i] for i in tidx]

            self.source_buffer = _sbuf
            self.target_buffer = _tbuf

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict.get(w, 1) for w in ss]
                if self.src_vocab_size > 0:
                    ss = [w if w < self.src_vocab_size else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                # append bos and eos to target
                tt = [2] + [self.target_dict.get(w, 1) for w in tt] + [3]
                if self.tgt_vocab_size > 0:
                    tt = [w if w < self.tgt_vocab_size else 1 for w in tt]

                if len(ss) > self.max_seq_length or \
                   len(tt) > self.max_seq_length:
                    continue

                source.append(ss)
                target.append(tt)
                batch_tokens += len(ss) + len(tt)

                if self.batch_tokens > 0:
                    if batch_tokens >= self.batch_tokens:
                        break
                else:
                    if len(source) >= self.batch_size:
                        break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target
