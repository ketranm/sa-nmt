#!/usr/bin/env bash
python train.py -datasets iwslt/train.en-de.de.tok.bpe iwslt/train.en-de.en.tok.bpe -valid_datasets iwslt/tst2014.en-de.de.tok.bpe iwslt/tst2014.en-de.en.tok.bpe -dicts iwslt/train.en-de.de.tok.bpe.pkl iwslt/train.en-de.en.tok.bpe.pkl -share_decoder_embeddings -word_vec_size 32 -rnn_size 32 -encoder_type sabrnn -encode_multi_key -share_attn -report_every 2 -learning_rate 0.001
