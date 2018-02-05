import argparse


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Model options
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=512,
                        help='Word embedding for both.')
    parser.add_argument('-share_decoder_embeddings', action='store_true',
                        help='Share the word and out embeddings for decoder.')

    # RNN Options
    parser.add_argument('-encoder_type', type=str, default='brnn',
                        choices=['brnn', 'sabrnn', 'fabrnn'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn'],
                        help='Type of decoder layer to use.')

    parser.add_argument('-layers', type=int, default=2,
                        help='Number of layers in enc/dec.')
    parser.add_argument('-rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    parser.add_argument('-rnn_type', type=str, default='LSTM',
                        choices=['LSTM'],
                        help="""The gate type to use in the RNNs""")
    parser.add_argument('-encode_multi_key', action='store_true',
                        help="""Using multi keys encoding of source""")
    parser.add_argument('-share_attn', action='store_true',
                        help="""sharing attention weights""")


def train_opts(parser):
    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    # GPU
    parser.add_argument('-gpuid', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=42,
                        help="""Random seed used for the experiments
                        reproducibility.""")
    parser.add_argument('-max_updates', type=int, default=1000000,
                        help="""max number of updates""")

    # Init options
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
    parser.add_argument('-epochs', type=int, default=13,
                        help='Number of training epochs')
    parser.add_argument('-optim', default='adam',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    # learning rate
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate. If adagrad/adadelta/adam
                        is used, then this is the global learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-max_decay_step', type=int, default=5,
                        help='maximum number of decay step for learning rate')
    parser.add_argument('-start_checkpoint_at', type=int, default=0,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")
    parser.add_argument('-start_eval_checkpoint_at', type=int, default=0,
                        help="""Start checkpointing every eval after and including
                        this eval""")
    parser.add_argument('-report_every', type=int, default=50,
                        help="Print stats at this interval.")
    parser.add_argument('-eval_every', type=int, default=10000,
                        help="Evaluate at this interval.")
    parser.add_argument('-min_thres', type=float, default=-5.0,
                        help="clip the value lower than this point.")
    parser.add_argument('-max_thres', type=float, default=7.0,
                        help="clip the value larger than this point.")
    parser.add_argument('-hard', action='store_true',
                        help="using hard structured attention.")


def preprocess_opts(parser):
    # Dictionary Options
    parser.add_argument('-src_vocab_size', type=int, default=-1,
                        help="Size of the source vocabulary")
    parser.add_argument('-tgt_vocab_size', type=int, default=-1,
                        help="Size of the target vocabulary")

    # Truncation options
    parser.add_argument('-max_seq_length', type=int, default=100,
                        help="Maximum sequence length")

    # Data processing options
    parser.add_argument('-shuffle', type=int, default=1,
                        help="Shuffle data")


def add_md_help_argument(parser):
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        usage_text = super(MarkdownHelpFormatter, self)._format_usage(
            usage, actions, groups, prefix)
        return '\n```\n%s\n```\n\n' % usage_text

    def format_help(self):
        self._root_section.heading = '# %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self).start_section('## **%s**' % heading)

    def _format_action(self, action):
        lines = []
        action_header = self._format_action_invocation(action)
        lines.append('### **%s** ' % action_header)
        if action.help:
            lines.append('')
            lines.append('```')
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
            lines.append('```')
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()
