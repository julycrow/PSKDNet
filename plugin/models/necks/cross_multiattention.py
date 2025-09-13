import torch
import torch.nn as nn
from mmdet.models import NECKS
# import rearrange
from einops import rearrange
import torch.nn.functional as F
import math


class PositionEncodingLearned2D(nn.Module):
    def __init__(self, dim_model, row_len=50, col_len=100, padding_idx=0):
        super(PositionEncodingLearned2D, self).__init__()
        self.row_embed = nn.Embedding(row_len, dim_model // 2)
        self.col_embed = nn.Embedding(col_len, dim_model // 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        n, _, h, w = x.shape
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_embed = self.col_embed(i)
        y_embed = self.row_embed(j)
        pos = torch.cat([x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)], dim=-1).permute(2,
                                                                                                                      0,
                                                                                                                      1).unsqueeze(
            0).repeat(n, 1, 1, 1)
        return pos


@NECKS.register_module()
class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads=8, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)
        # self.positional_encoding = self.create_positional_encoding(50, 100, emb_dim)
        self.pe = PositionEncodingLearned2D(emb_dim)

    def create_positional_encoding(self, height, width, emb_dim):
        pe = torch.zeros(emb_dim, height, width)
        y_pos = torch.arange(0, height).unsqueeze(1).float()
        x_pos = torch.arange(0, width).unsqueeze(0).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pe[0::2, :, :] = torch.sin(x_pos * div_term.unsqueeze(1))
        pe[1::2, :, :] = torch.cos(y_pos * div_term.unsqueeze(0))
        return pe.unsqueeze(0)

    def init_weights(self):
        # 跳过
        pass
        # nn.init.xavier_normal_(self.Wq.weight)
        # nn.init.xavier_normal_(self.Wk.weight)
        # nn.init.xavier_normal_(self.Wv.weight)

    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        x = x.unsqueeze(0).contiguous()  # torch.Size([256, 50, 100]) ->  # torch.Size([1, 256, 50, 100])
        b, c, h, w = x.shape  # torch.Size([1, 256, 50, 100])
        x = x + self.pe(x[:, :, :h, :w])  # Add positional encoding
        x = self.proj_in(x)  # torch.Size([1, 256, 50, 100])


        x = rearrange(x, 'b c h w -> b (h w) c')  # [batch_size, h*w, c] = [1, 5000, 256]
        context = context.unsqueeze(0).contiguous()
        context = context + self.pe(context[:, :, :h, :w])  # Add positional encoding
        context = self.proj_in(context)
        context = rearrange(context, 'b c h w -> b (h w) c')
        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [1, 5000, 256]
        K = self.Wk(context)  # [batch_size, seq_len, emb_dim] = [1, 5000, 256]
        V = self.Wv(context)

        Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.emb_dim)  # [batch_size, h*w, emb_dim]

        # print(out.shape)
        #
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [batch_size, c, h, w]
        out = self.proj_out(out)  # [batch_size, c, h, w]
        out = out.squeeze(0).contiguous()  # torch.Size([1, 256, 50, 100])->torch.Size([256, 50, 100])

        # out = rearrange(out, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]
        return out


# x = torch.rand((2, 256, 50, 100))
# pe = PositionEncodingLearned2D(256)
# x = pe(x)
