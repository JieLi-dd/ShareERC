import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding

class EmoAudio2(nn.Module):

    def __init__(self, embed_dim, num_heads, layers, attn_dropout, gelu_dropout, embed_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.position_embed = SinusoidalPositionalEmbedding(embed_dim)
        self.embed_dropout = nn.Dropout(embed_dropout, inplace=True)
        self.embed_scale = embed_dim ** -0.5    # 为了添加位置信息所占的比例比较小，对原来的信息进行放大的比例

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = 0


    def forward(self, audio_in_q, text_in_k, text_in_v):
        """

        :param audio_in_q: (batch, src_len, embed_dim)
        :param text_in_k:  (batch, src_len, embed_dim)
        :param text_in_v:  (batch, src_len, embed_dim)
        :return:
        """
        audio_q = self.embed_scale * audio_in_q
        if self.position_embed is not None:
            audio_q += self.position_embed(audio_in_q)
        audio_q = self.embed_dropout(audio_q[:, :, 0])
        audio_q = self.embed_dropout(audio_q)

        if text_in_k is not None and text_in_v is not None:
            text_k = self.embed_scale * text_in_k
            text_v = self.embed_scale * text_in_v
            if self.position_embed is not None:
                text_k += self.position_embed(text_in_k[:, :, 0])
                text_v += self.position_embed(text_in_v[:, :, 0])
            text_v = self.embed_dropout(text_v)
            text_k = self.embed_dropout(text_k)

        return None



class TransformerEnconderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, gelu_dropout=0.1, res_dropout=0.0, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim,
        self.num_heads = num_heads

        self.cross_model = 0

    def forward(self):
        return None


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "输入维度：（batch_size, head, len, head_dim"
    feat_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(feat_dim)
    if mask is not None:
        mask = mask.transpose(-2, -1).expand_as(scores)
        scores = scores.masked_fill(mask==0, -1e4)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.w * (mean - x) / (std + self.eps) + self.b


class Generator(nn.Module):
    """
    定义一个线性映射层 + softmax函数层
    """

    def __init__(self, d_model, text_dim_or_audio_dim):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, text_dim_or_audio_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.proj(x))


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EmoAudio(nn.Module):
    """
    音频特征 = 文本特征 + 心情特征
    本模块致力于提取心情特征
    """
    def __init__(self, text_encoder, audio_encoder):
        super(EmoAudio, self).__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

    def forward(self, text, audio, text_mask, audio_mask):
        return self.audio_encode(self.text_encode(text, text_mask), text_mask, audio, audio_mask)

    def text_encode(self, text, text_mask):
        return self.text_encoder(text, text_mask)

    def audio_encode(self, memory, text_mask, audio, audio_mask):
        return self.audio_encoder(audio, memory, text_mask, audio_mask)

    @staticmethod
    def make_model(N=5, feat_dim=768, ff_dim=748*4, head=12, dropout=0.1):
        c = copy.deepcopy
        attn = MultiHeadAttention(head=head, feat_dim=feat_dim, dropout=dropout)
        ff = PositionwiseFeedForward(feat_dim=feat_dim, ff_dim=ff_dim, dropout=dropout)
        model = EmoAudio(
            TextEncoder(TextEnconderLayer(feat_dim, c(attn), c(ff), dropout), N),
            AudioEnconder(AudioEnconderLayer(feat_dim, c(attn), c(attn), c(ff), dropout), N)
        )

        model.cpu()
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        model.cuda()
        return model


class TextEncoder(nn.Module):

    def __init__(self, layer, N):
        super(TextEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, text, text_mask):
        middle_results = []
        for layer in self.layers:
            text = layer(text, text_mask)
            text = self.norm(text)
            middle_results.append(text)
        return middle_results



class TextEnconderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TextEnconderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, text, text_mask):
        self.sublayer[0](text, lambda text: self.self_attn(text, text, text, text_mask))
        return self.sublayer[1](text, self.feed_forward)


class AudioEnconder(nn.Module):

    def __init__(self, layer, N):
        super(AudioEnconder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, audio, memory, text_mask, audio_mask):
        '''
        for i, layer in enumerate(self.layers):
            audio = layer(audio, memory[i], text_mask, audio_mask)

        '''
        for i, layer in enumerate(self.layers):
            # audio = layer(audio, memory[i], text_mask, audio_mask)
            audio = layer(audio, memory[-1], text_mask, audio_mask)
            audio = self.norm(audio)
        return audio


class AudioEnconderLayer(nn.Module):

    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(AudioEnconderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, audio, memory, text_mask, audio_mask):
        m = memory
        audio = self.sublayer[0](audio, lambda audio: self.self_attn(audio, audio, audio, audio_mask))
        audio = self.sublayer[1](audio, lambda audio: self.cross_attn(audio, m, m, text_mask, is_cross_attn=True))
        return self.sublayer[2](audio, self.feed_forward)



class MultiHeadAttention(nn.Module):

    def __init__(self, head, feat_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert feat_dim % head == 0
        self.head_dim = feat_dim // head
        self.head = head
        self.linears = clones(nn.Linear(feat_dim, feat_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, is_cross_attn=False):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)
        batch_size = query.size(0)

        query, key, value = [
            lin(x).view(batch_size, -1, self.head, self.head_dim).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = (x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.head_dim))
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, feat_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.f1 = nn.Linear(feat_dim, ff_dim)
        self.f2 = nn.Linear(ff_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.f2(self.dropout(self.f1(x).relu()))


if __name__ == '__main__':
    emoaudio = EmoAudio.make_model()
    text = torch.randn((1,38,768))
    audio = torch.randn((1,130,768))
    text_mask = torch.ones((1,38))
    audio_mask = torch.ones((1, 130))
    out = emoaudio(text, audio, text_mask, audio_mask)
    print(out)