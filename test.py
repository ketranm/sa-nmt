import torch
from torch.autograd import Variable
import torch.nn as nn


# Structured Attention and our models
class MatrixTree(nn.Module):
    """Implementation of the matrix-tree theorem for computing marginals
    of non-projective dependency parsing. This attention layer is used
    in the paper "Learning Structured Text Representations."
    """
    def __init__(self, eps=1e-5):
        self.eps = eps
        super(MatrixTree, self).__init__()

    def forward(self, input, lengths=None):
        """
        Args:
            input: FloatTensor of size (batch, n, n)
            lengths: a list of real lengths [10, 9, 8, ...]
        Returns:
            a FloatTensor (batch, n, n): attention distribution
        """
        laplacian = input.exp()
        output = input.clone()
        output.data.fill_(0)
        for b in range(input.size(0)):
            lx = lengths[b] if lengths is not None else input.size(1)
            input_b = input[b, :lx, :lx]
            lap = laplacian[b, :lx, :lx].masked_fill(
                Variable(torch.eye(lx).ne(0)), 0)
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


matrix_tree = MatrixTree()

batch, n = 2, 4
x = Variable(torch.randn(batch, n, n))
lengths = [4, 2]
out = matrix_tree(x, lengths)
out = out.transpose(1, 2)
print('attention matrix')
print(out[0])
y = out[0]
for i in range(y.size(0)):
    for j in range(y.size(1)):
        print(y[i, j].data[0])
