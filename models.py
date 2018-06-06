import torch
import torch.nn as nn
from torch.autograd import Variable
from attention import GlobalAttention, SelfAttention
from Utils import aeq
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import math


class EncoderBase(nn.Module):
    """
    EncoderBase class for sharing code among various encoder.
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch = input.size()
        if lengths is not None:
            n_batch_ = len(lengths)  # lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        raise NotImplementedError


class Encoder(EncoderBase):
    """ The standard RNN encoder. """
    def __init__(self, word_vocab_size, embedding_size, word_padding_idx,
                 num_layers, hidden_size, dropout, bidirectional=True):
        super(Encoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = nn.Embedding(word_vocab_size,
                                       embedding_size,
                                       padding_idx=word_padding_idx)

        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        self.word_embs = emb  # for accessing later

        packed_emb = emb
        if lengths is not None:
            packed_emb = pack(emb, lengths)

        outputs, hidden = self.rnn(packed_emb, hidden)

        if lengths is not None:
            outputs = unpack(outputs)[0]

        return outputs, hidden


# Structured Attention and our models
class MatrixTree(nn.Module):
    """Implementation of the matrix-tree theorem for computing marginals
    of non-projective dependency parsing. This attention layer is used
    in the paper "Learning Structured Text Representations."
    """
    def __init__(self, eps=1e-5, device=torch.device('cpu')):
        self.eps = eps
        super(MatrixTree, self).__init__()
        self.device = device

    def forward(self, input, lengths=None):
        laplacian = input.exp()
        output = input.clone()
        output.data.fill_(0)
        for b in range(input.size(0)):
            lx = lengths[b] if lengths is not None else input.size(1)
            input_b = input[b, :lx, :lx]
            lap = laplacian[b, :lx, :lx].masked_fill(
                torch.eye(lx).to(self.device).ne(0), 0)
            lap = -lap + torch.diag(lap.sum(0))
            # store roots on diagonal
            lap[0] = input_b.diag().exp()
            inv_laplacian = lap.inverse()

            factor = inv_laplacian.diag().unsqueeze(1)\
                                         .expand(lx, lx).transpose(0, 1)
            term1 = input_b.exp().mul(factor).clone()
            term2 = input_b.exp().mul(inv_laplacian.transpose(0, 1)).clone()
            term1[:, 0] = 0
            term2[0] = 0
            output_b = term1 - term2
            roots_output = input_b.diag().exp().mul(
                inv_laplacian.transpose(0, 1)[0])
            output[b, :lx, :lx] = output_b + torch.diag(roots_output)
        return output


class TreeAttention(nn.Module):
    """Structured attention class"""
    def __init__(self, dim, min_thres=-5, max_thres=7, hard=False,
                 device=torch.device('cpu')):
        super(TreeAttention, self).__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.root_query = nn.Parameter(torch.randn(dim))
        self.scale = math.sqrt(1 / dim)
        self.dtree = MatrixTree()
        self.min_thres = min_thres
        self.max_thres = max_thres
        self.hard = hard
        self.device = device

    def forward(self, input, punct_mask=None, lengths=None):
        s_len, batch, dim = input.size()
        input = input.contiguous().transpose(0, 1) \
            .contiguous().view(-1, dim)

        q = self.q(input).view(batch, s_len, -1)  # (batch, s_len, dim)
        k = self.k(input).view(batch, s_len, -1)  # (batch, s_len, dim)
        v = self.v(input).view(batch, s_len, -1)  # (batch, s_len, dim)
        _score = torch.bmm(q, k.transpose(1, 2))  # (batch, s_len, s_len)
        # compute root
        r_ = self.root_query.view(1, -1, 1).expand(batch, dim, 1)
        root = torch.bmm(k, r_).squeeze(-1)  # (batch, s_len)
        mask = torch.eye(s_len).to(self.device)
        score = _score.clone()
        for b in range(batch):
            score[b] = _score[b] * mask + torch.diag(root[b])

        # normalized
        score *= self.scale
        if punct_mask is not None:
            punct_mask = punct_mask.transpose(0, 1)
            punct_mask = punct_mask[:, None, :].expand(batch, s_len, s_len) \
                .transpose(1, 2)
            score.data.masked_fill_(punct_mask, -math.inf)
        score = score.clamp(self.min_thres, self.max_thres)
        self.edge_score = score.transpose(1, 2)
        edge_score = self.dtree(score, lengths).transpose(1, 2)
        # edge_score.sum(2) == 1
        if self.hard:
            y = edge_score.data.new(edge_score.size()).fill_(0)
            _, max_idx = edge_score.data.max(2)
            y.scatter_(2, max_idx[:, :, None], 1)
            hard_edge = (Variable(y) - edge_score).detach() + edge_score
            edge_score = hard_edge

        return torch.bmm(edge_score, v)


class SAEncoder(Encoder):
    """ The structured attention RNN encoder. """
    def __init__(self, word_vocab_size, embedding_size, word_padding_idx,
                 num_layers, hidden_size, dropout, bidirectional=True,
                 encode_multi_key=True, min_thres=-5, max_thres=7, hard=False):
        super(SAEncoder, self).__init__(word_vocab_size, embedding_size,
                                        word_padding_idx,
                                        num_layers, hidden_size, dropout,
                                        bidirectional=True)
        self.tree_attn = TreeAttention(hidden_size, min_thres, max_thres, hard)
        self.encode_multi_key = encode_multi_key
        if not self.encode_multi_key:
            self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.punct_idx = None

    def punct(self, punct_idx):
        self.punct_idx = punct_idx

    def forward(self, input, lengths=None):
        outputs, hidden = Encoder.forward(self, input, lengths)
        # find puncts
        punct_mask = None
        if self.punct_idx is not None:
            punct = set(input.data.contiguous().view(-1).tolist())
            punct &= self.punct_idx
            if len(punct) > 0:
                punct_mask = 0
                for p in punct:
                    punct_mask += input.data.eq(p)

        tree_outputs = self.tree_attn(outputs, punct_mask, lengths)
        # tree_outputs has size (batch_size, s_len, hidden_size)
        if not self.encode_multi_key:
            # compute gate syntax
            z = self.linear(outputs.view(-1, outputs.size(2))).sigmoid()
            gtree = z.view_as(outputs) * tree_outputs.transpose(0, 1)
            outputs = outputs + gtree
            return outputs, hidden
        return (outputs, tree_outputs), hidden


class FAEncoder(Encoder):
    """ The flat attention RNN encoder. """
    def __init__(self, word_vocab_size, embedding_size, word_padding_idx,
                 num_layers, hidden_size, dropout, bidirectional=True,
                 encode_multi_key=True):
        super(FAEncoder, self).__init__(word_vocab_size, embedding_size,
                                        word_padding_idx,
                                        num_layers, hidden_size, dropout,
                                        bidirectional=True)
        self.attn = SelfAttention(hidden_size)
        self.encode_multi_key = encode_multi_key

    def punct(self, punct_idx):
        self.punct_idx = punct_idx

    def forward(self, input, lengths=None):
        mask = input.data.eq(0).t()
        outputs, hidden = Encoder.forward(self, input, lengths)
        punct_mask = None
        if self.punct_idx is not None:
            punct = set(input.data.contiguous().view(-1).tolist())
            punct &= self.punct_idx
            if len(punct) > 0:
                punct_mask = 0
                for p in punct:
                    punct_mask += input.data.eq(p)
        self_attn_outputs = self.attn(outputs, mask, punct_mask)
        if not self.encode_multi_key:
            outputs = outputs + self_attn_outputs.transpose(0, 1)
            return outputs, hidden
        return (outputs, self_attn_outputs), hidden


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):
    def __init__(self, word_vocab_size, embedding_size, word_padding_idx,
                 num_layers, hidden_size, dropout, multi_key=False,
                 shared_attn=False):

        input_size = embedding_size + hidden_size
        self.hidden_size = hidden_size

        super(Decoder, self).__init__()
        self.embeddings = nn.Embedding(word_vocab_size, embedding_size,
                                       padding_idx=word_padding_idx)
        self.rnn = StackedLSTM(num_layers, input_size, hidden_size, dropout)
        self.attn = GlobalAttention(hidden_size, multi_key, shared_attn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context, mask=None, init_output=None):
        emb = self.embeddings(input)
        batch_size = input.size(1)
        h_size = (batch_size, self.hidden_size)

        outputs = []
        if init_output is None:
            output = Variable(emb.data.new(*h_size).zero_(),
                              requires_grad=False)
        else:
            output = init_output
        attns = []

        # set mask
        if mask is not None:
            self.attn.apply_mask(mask)

        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(rnn_output,
                                          context)
            output = self.dropout(attn_output)
            outputs += [output]
            attns.append(attn)
        attns = torch.stack(attns)
        outputs = torch.stack(outputs)
        return outputs, hidden, attns


class NMT(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

    def _init_hidden(self, enc_hidden):
        hidden = (self._fix_enc_hidden(enc_hidden[0]),
                  self._fix_enc_hidden(enc_hidden[1]))
        return hidden

    def forward(self, src, tgt, src_lengths):
        context, enc_hidden = self.encoder(src, src_lengths)
        if isinstance(context, tuple):
            context_ = (context[0].transpose(0, 1), context[1])
        else:
            context_ = context.transpose(0, 1)
        enc_hidden = self._init_hidden(enc_hidden)
        src_pad_mask = src.data.eq(0).t()
        out, dec_hidden, attn = self.decoder(tgt, enc_hidden,
                                             context_, src_pad_mask)
        return out


def make_encoder(opt):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
    """
    if opt.encoder_type == "sabrnn":
        return SAEncoder(opt.src_vocab_size, opt.word_vec_size, 0,
                         opt.layers, opt.rnn_size, opt.dropout,
                         bidirectional=True,
                         encode_multi_key=opt.encode_multi_key,
                         min_thres=opt.min_thres, max_thres=opt.max_thres,
                         hard=opt.hard)
    elif opt.encoder_type == "fabrnn":
        return FAEncoder(opt.src_vocab_size, opt.word_vec_size, 0,
                         opt.layers, opt.rnn_size, opt.dropout,
                         bidirectional=True,
                         encode_multi_key=opt.encode_multi_key)
    else:
        return Encoder(opt.src_vocab_size, opt.word_vec_size, 0,
                       opt.layers, opt.rnn_size, opt.dropout)


def make_decoder(opt):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    return Decoder(opt.tgt_vocab_size, opt.word_vec_size, 0,
                   opt.layers, opt.rnn_size, opt.dropout,
                   opt.encode_multi_key, opt.share_attn)


def make_base_model(model_opt, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu: Boolean: whether to use gpu.
        checkpoint: the snapshot model.
    Returns:
        the NMTModel.
    """
    # Make encoder.
    encoder = make_encoder(model_opt)
    decoder = make_decoder(model_opt)

    # Make NMT (= encoder + decoder).
    model = NMT(encoder, decoder)

    # Make Generator.
    generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, model_opt.tgt_vocab_size),
        nn.LogSoftmax(dim=-1))
    if model_opt.share_decoder_embeddings:
        generator[0].weight = decoder.embeddings.weight

    model.generator = generator
    # Load the model states from checkpoint.
    if checkpoint is not None:
        print('Loading model')
        model.load_state_dict(checkpoint['model'])
        model.dicts = checkpoint['dicts']

    return model
