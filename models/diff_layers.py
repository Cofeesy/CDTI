import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def Attn_1(heads=8, layers=1, channels=64):
    encoder_layer = TransformerEncoderLayer_QKV(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return TransformerEncoder_QKV(encoder_layer, num_layers=layers)

class Attn_2(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0., is_fusion=False):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        self.is_fusion = is_fusion

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, itp_x=None, itp_time=None, itp_feature=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k
        
        if self.is_fusion:
            v_len = n if itp_feature is None else itp_feature.shape[1]
            q_input = x if itp_time is None else itp_time
            k_input = x if itp_feature is None else itp_feature
        else:
            v_len = n if itp_x is None else itp_x.shape[1]
            q_input = x if itp_x is None else itp_x
            k_input = x if itp_x is None else itp_x
        
        assert v_len == self.seq_len, f'the sequence length of the values must be {self.seq_len} - {v_len} given'

        queries = self.to_q(q_input)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        v_input = x
        keys = self.to_k(k_input)
        values = self.to_v(v_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class Fusion_2(Attn_2):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__(dim, seq_len, k, heads, dim_head, one_kv_head, share_kv, dropout, is_fusion=True)

    def forward(self, x, itp_time=None, itp_feature=None, **kwargs):
        return super().forward(x, itp_time=itp_time, itp_feature=itp_feature, **kwargs)

class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)

    def forward(self, query, key, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(query, key, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder_QKV(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(query, key, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class TemporalLearning(nn.Module):
    def __init__(self, channels, nheads, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.time_layer = Attn_1(heads=nheads, layers=1, channels=channels)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)

    def forward(self, y, base_shape, itp_y=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y.permute(2, 0, 1)
        if self.is_cross:
            itp_y = itp_y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            q = itp_y.permute(2, 0, 1)
            y = self.time_layer(q, q, v).permute(1, 2, 0)
        else:
            y = self.time_layer(v, v, v).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y
    
class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels, features):
        super(SimpleCNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(),
            # Block 2
            nn.Conv2d(features[0], features[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU()
            # ...可以继续添加更多卷积块
        )

    def forward(self, x):
        return self.encoder(x)


class Fusion1Learning(nn.Module):
    def __init__(self, channels, nheads, is_cross=False):
        super().__init__()
        self.is_cross = is_cross
        self.time_layer = Attn_1(heads=nheads, layers=1, channels=channels)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)

    def forward(self, y, time, feature, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y.permute(2, 0, 1)

        if self.is_cross:
            time = time.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            feature = feature.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

            q = time.permute(2, 0, 1)
            k = feature.permute(2, 0, 1)
            y = self.time_layer(q, k, k).permute(1, 2, 0)
        else:
            y = self.time_layer(v, v, v).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

class FeatureLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, proj_t=16, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.attn = Attn_2(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_attn = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, itp_y=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_cross:
            itp_y_attn = itp_y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
            y_attn = self.attn(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        y_attn = self.norm1_attn(y_attn)

        return y_attn

class Fusion2Learning(nn.Module):
    def __init__(self, channels, nheads, target_dim, proj_t=16, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.attn = Fusion_2(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_attn = nn.GroupNorm(4, channels)

    def forward(self, y, time, feature, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_cross:
            itp_time_attn = time.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
            itp_feature_attn = feature.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
            y_attn = self.attn(y_attn.permute(0, 2, 1), itp_time_attn.permute(0, 2, 1), itp_feature_attn.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        y_attn = self.norm1_attn(y_attn)

        return y_attn

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        assert channels % 2 == 0
        self.channel = channels // 2

        self.input_conv = nn.Conv2d(channels, self.channel, kernel_size=1)
        self.context_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.final_conv = nn.Conv2d(self.channel, channels, kernel_size=1)

    def forward(self, x):
        batch, channels, height, width = x.size()

        input_x = self.input_conv(x)
        input_x = input_x.view(batch, self.channel, height * width)

        context_mask = self.context_conv(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)

        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.view(batch, self.channel, 1, 1)
        context = self.final_conv(context)

        mask_ch = self.sigmoid(context)
        return x * mask_ch

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        assert channels % 2 == 0
        self.channel = channels // 2

        self.g_conv = nn.Conv2d(channels, self.channel, kernel_size=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.theta_conv = nn.Conv2d(channels, self.channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, height, width = x.size()

        g_x = self.g_conv(x)
        avg_x = self.avg_pool(g_x)
        avg_x = self.softmax(avg_x.view(batch, self.channel))
        avg_x = avg_x.view(batch, self.channel, 1)

        theta_x = self.theta_conv(x)
        theta_x = theta_x.view(batch, self.channel, height * width)
        theta_x = theta_x.permute(0, 2, 1)

        context = torch.bmm(theta_x, avg_x)
        context = context.view(batch, height * width)
        mask_sp = self.sigmoid(context)
        mask_sp = mask_sp.view(batch, 1, height, width)

        return x * mask_sp

class SimpleDecoder(nn.Module):
    def __init__(self, in_dim, time_frames, out_channels, out_len):
        super(SimpleDecoder, self).__init__()
        self.feature_decoder = nn.Linear(in_dim, out_channels)
        self.out_len = out_len

    def forward(self, x):
        x = self.feature_decoder(x)
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=self.out_len, mode='linear', align_corners=False)
        return x

class FSAtnn(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()

        self.feature_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention(channels)

        self.channels = channels
        self.num_heads = num_heads

    def get_positional_encoding(self, seq_len, d_model, device):
        pe = torch.zeros(1, seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe.to(device)

    def forward(self, x):
        B, C, F, T = x.shape
        transformer_dim = C * F

        x_enhanced = self.spatial_attention(self.feature_attention(x))
        x_seq = x_enhanced.permute(0, 3, 1, 2).flatten(start_dim=2)

        pos_embed = self.get_positional_encoding(T, transformer_dim, x.device)
        x_with_pos = x_seq + pos_embed

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=self.num_heads, 
            batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(x.device)
        transformer_out = transformer_encoder(x_with_pos)

        decoder = SimpleDecoder(
            in_dim=transformer_dim, 
            time_frames=T, 
            out_channels=C, 
            out_len=48
        ).to(x.device)
        output = decoder(transformer_out)

        return output

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Linear(1, embedding_dim)
        self.pos_emb = nn.Embedding(num_embeddings=maxlen, embedding_dim=embedding_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x_emb = self.token_emb(x)
        positions = torch.arange(0, x.size(1), device=x.device)
        pos_emb = self.pos_emb(positions)
        return x_emb + pos_emb
