from __future__ import print_function
import numpy
import pickle as pkl
import sys

from collections import OrderedDict


def main():
    for filename in sys.argv[1:]:
        print('Processing', filename)
        word_freqs = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['<pad>'] = 0
        worddict['<unk>'] = 1
        worddict['<bos>'] = 2
        worddict['<eos>'] = 3
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii+4

        with open('%s.pkl' % filename, 'wb') as f:
            pkl.dump(worddict, f)

        print('Done')


if __name__ == '__main__':
    main()
