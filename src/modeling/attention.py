import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

import numpy as np


class SimpleDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(SimpleDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(dim))
        self.query = torch.nn.Linear(dim, 1)

    def forward(self, X):
        return (self.query(X) / self.scale).softmax(-2)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn_logits = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)

        attn_logits = self.dropout(F.softmax(attn_logits, dim=-1))
        attn_value = torch.matmul(attn_logits, v)

        return attn_value, attn_logits


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, qk_dim, v_dim, num_heads, num_classes, drop_out=0.0):
        super(MultiheadAttention, self).__init__()

        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.drop_out = drop_out
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qk_proj = nn.Sequential(
            nn.Linear(input_dim, 2 * self.qk_dim * num_heads, bias=False),
            nn.Dropout(p=self.drop_out)
        )
        self.v_proj = nn.Sequential(
            nn.Linear(input_dim, self.v_dim * num_heads, bias=False),
            nn.Dropout(p=self.drop_out)
        )
        self.o_proj = nn.Sequential(
            nn.Linear(self.v_dim * num_heads, self.v_dim * num_heads, bias=False),
            nn.Dropout(p=self.drop_out)
        )
        self.scaled_dot = ScaledDotProductAttention(temperature=qk_dim**0.5)

        self.linears = torch.nn.Sequential(
            torch.nn.Linear(self.v_dim * num_heads, v_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_out),
            torch.nn.Linear(v_dim, num_classes),
            # torch.nn.Softmax(dim=1)
        )
        # self._reset_parameters()

    # def _reset_parameters(self):
    #     # Original Transformer initialization, see PyTorch documentation
    #     nn.init.xavier_uniform_(self.qkv_proj.weight)
    #     self.qkv_proj.bias.data.fill_(0)
    #     nn.init.xavier_uniform_(self.o_proj.weight)
    #     self.o_proj.bias.data.fill_(0)

    def _forward_attn(self, x, mask=None, return_attention=False):
        batch_size, seq_length, token_dim = x.size()
        qk = self.qk_proj(x)
        v = self.v_proj(x)

        # Separate Q, K from linear output
        qk = qk.reshape(batch_size, seq_length, self.num_heads, 2 * self.qk_dim)
        qk = qk.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k = qk.chunk(2, dim=-1)

        # Separate V heads
        v = v.reshape(batch_size, seq_length, self.num_heads, self.v_dim)
        v = v.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]

        # Determine value outputs
        attn_value, attn_logits = self.scaled_dot(q, k, v, mask=mask)
        attn_value = attn_value.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        attn_value = attn_value.reshape(
            batch_size, seq_length, self.v_dim * self.num_heads
        )
        o = self.o_proj(attn_value)

        if return_attention:
            return o, attn_logits
        else:
            return o

    # def forward(self, x: Tensor, corelen, return_attention=False, *args):
        # corelen_start = corelen
        # corelen_end = np.append(corelen[1:].detach().cpu(), None)

        # outputs = []
        # attentions = []
        # for i, j in zip(corelen_start, corelen_end):
        #     attn_outs, attn_logits = self._forward_impl(
        #         x[None, i:j, ...], return_attention=True
        #     )
        #     outs = self.linears(attn_outs.squeeze(0))
        #     outs = torch.mean(outs, dim=0)
        #     outputs.append(outs)
        #     attentions.append(attn_logits)

        # if return_attention:
        #     return torch.stack(outputs, dim=0), torch.stack(attentions, dim=0)
        # else:
            # return torch.stack(outputs, dim=0)

    def forward(self, x: Tensor, return_attention=False, *args):
        attn_outs, attn_logits = self._forward_attn(x, return_attention=True) # [batch, seq_len, head*v_dim]
        
        outs = torch.mean(attn_outs, dim=1) #[batch, head*v_dim]
        outs = self.linears(outs) #[batch, num_classes]

        if return_attention:
            return outs, attn_logits
        else:
            return outs

# This module is copied from code of "Attention-based Deep Multiple Instance Learning"
class AttentionMIL(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AttentionMIL, self).__init__()
        self.L = input_dim
        self.num_classes = num_classes
        self.D = 128
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
        )

    def _forward_impl(self, x):
        x = x.squeeze(0)

        H = x  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob  # , Y_hat, A

    def forward(self, x: Tensor, corelen, *args):
        corelen_start = corelen
        corelen_end = np.append(corelen[1:].detach().cpu(), None)

        outputs = []
        for i, j in zip(corelen_start, corelen_end):
            attn_outs = self._forward_impl(x[None, i:j, ...])
            outputs.append(attn_outs)

        return torch.cat(outputs, dim=0)
