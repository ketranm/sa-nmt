import torch
import torch.nn as nn
from Utils import aeq
import math
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self attention class"""
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(1 / dim)

    def forward(self, input, mask=None, punct_mask=None):
        """
        input (FloatTensor): s_len, batch, dim
        mask (ByteTensor): batch x s_len
        """
        s_len, batch, dim = input.size()
        input = input.contiguous().transpose(0, 1) \
            .contiguous().view(-1, dim)

        q = self.q(input).view(batch, s_len, -1)
        k = self.k(input).view(batch, s_len, -1)
        v = self.v(input).view(batch, s_len, -1)
        score = torch.bmm(q, k.transpose(1, 2)) * self.scale
        # now (batch, s_len, s_len) FloatTensor
        if mask is not None:
            mask = mask[:, None, :].expand(batch, s_len, s_len)
            score.data.masked_fill_(mask, -math.inf)
        if punct_mask is not None:
            punct_mask = punct_mask.transpose(0, 1)  # batch, s_len
            punct_mask = punct_mask[:, None, :].expand(batch, s_len, s_len)
            score.data.masked_fill_(punct_mask, -math.inf)
        # need proper masking
        attn = F.softmax(score.view(-1, s_len), dim=-1).view(-1, s_len, s_len)
        self.score = attn
        return torch.bmm(attn, v)


class GlobalAttention(nn.Module):
    """
    Luong Attention. This implement general attention
    Concrete distribution: The Concrete Distribution: A Continuous Relaxation
                            of Discrete Random Variables
    """
    def __init__(self, dim, multi_key=False, share_attn=False):
        """
        dim (Int): dimension of input vector
        multi_key (Boolean): using multi keys encoder
        share_attn (Boolean): sharing attention weights between
            semantic and syntactic annotations
        """
        super(GlobalAttention, self).__init__()
        # make a local copy of hyper-parameters
        self.dim = dim
        self.share_attn = share_attn
        self.multi_key = multi_key

        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.share_attn = share_attn
        if multi_key:
            if not share_attn:  # using a separate attention
                self.linear_sa = nn.Linear(dim, dim, bias=False)
            self.linear_out = nn.Linear(dim*3, dim, bias=False)
            self.gate = nn.Linear(dim, dim)
        else:
            self.linear_out = nn.Linear(dim*2, dim, bias=False)

        self.mask = None

    def apply_mask(self, mask):
        self.mask = mask

    def score(self, h_t, h_s, sa_attn=False):
        """
        h_t (FloatTensor): batch x tgt_len x dim
        h_s (FloatTensor): batch x src_len x dim
        returns scores (FloatTensor): batch x tgt_len x src_len:
            raw attention scores for each src index
        sa_attn (Boolean): using a separate attention for syntax context
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
        if sa_attn:
            h_t_ = self.linear_sa(h_t_)
        else:
            h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def forward(self, input, context):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output.
        context (FloatTensor): batch x src_len x dim: src hidden states
        """

        # one step input
        if isinstance(context, tuple):
            context, tree_context = context

        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # compute attention scores, as in Luong et al.
        align = self.score(input, context)

        if self.mask is not None:
            mask_ = self.mask[:, None, :]
            align.data.masked_fill_(mask_, -math.inf)

        # Softmax to normalize attention weights
        align_vectors = F.softmax(align, dim=-1)
        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)
        if self.multi_key:
            # sharing attention weight
            if self.share_attn:
                sc = torch.bmm(align_vectors, tree_context)
            else:
                # computing attention scores for syntax
                tree_align = self.score(input, tree_context, True)
                if self.mask is not None:
                    tree_align.data.masked_fill_(self.mask[:, None, :],
                                                 -math.inf)
                tree_align_vectors = F.softmax(tree_align, dim=-1)
                sc = torch.bmm(tree_align_vectors, tree_context)

            z = F.sigmoid(self.gate(input))  # batch x tgt_len x dim
            self.z = z  # for visualization
            sc = sc * z
            concat_c = torch.cat([c, input, sc], 2).view(batch*targetL, dim*3)
        else:
            concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)

        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        attn_h = F.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        return attn_h, align_vectors
