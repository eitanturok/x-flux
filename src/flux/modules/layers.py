import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from ..math import attention, rope
import torch.nn.functional as F

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):
        print('2' * 30)

        qkv = attn.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x

class LoraFluxAttnProcessor(nn.Module):

    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight


    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x) + self.qkv_lora(x) * self.lora_weight
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x) + self.proj_lora(x) * self.lora_weight
        print('1' * 30)
        print(x.norm(), (self.proj_lora(x) * self.lora_weight).norm(), 'norm')
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    def forward():
        pass


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class DoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class IPDoubleStreamBlockProcessor(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def __call__(self, attn, img, txt, vec, pe, image_proj, ip_scale=1.0, **attention_kwargs):

        # Prepare image for attention
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # IP-adapter processing
        ip_query = img_q  # latent sample query
        ip_key = self.ip_adapter_double_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_double_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value,
            dropout_p=0.0,
            is_causal=False
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)

        img = img + ip_scale * ip_attention

        return img, txt

class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
    ) -> tuple[Tensor, Tensor]:
        if image_proj is None:
            return self.processor(self, img, txt, vec, pe)
        else:
            return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)

# class ReRopeDoubleStreamBlock(DoubleStreamBlock):
#     def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
#         super().__init__(hidden_size, num_heads, mlp_ratio, qkv_bias=qkv_bias)

#     def shrink_img(self, img, current_height, current_width, idxs, target_width):
#         img = rearrange(img, "bs (h w) z -> bs z h w", h=current_height, w=current_width)
#         img = torch.index_select(img, -1, idxs)
#         img = rearrange(img, "bs z h w -> bs (h w) z", h=current_height, w=target_width)
#         return img

#     def shrink_pe(self, pe, txt_len, current_height, current_width, idxs, target_width):
#         txt_pe = pe[:, :, :txt_len, :, :, :]  # (bs, 1, txt_len, pe_dim//2, 2, 2)
#         img_pe = pe[:, :, txt_len:, :, :, :]  # (bs, 1, h_2*w_2, pe_dim//2, 2, 2)
#         img_pe = rearrange(img_pe, "bs j (h w) pe_dim k l -> bs j pe_dim k l h w", h=current_height, w=current_width)
#         img_pe = torch.index_select(img_pe, -1, idxs)
#         img_pe = rearrange(img_pe, "bs j pe_dim k l h w ->bs j (h w) pe_dim k l", h=current_height, w=target_width)
#         pe = torch.cat((txt_pe, img_pe), dim=2)
#         return pe

#     def forward(
#         self,
#         img: Tensor,
#         txt: Tensor,
#         vec: Tensor,
#         pe: Tensor,
#         txt_len: int,
#         current_height: int,
#         current_width: int,
#         target_height: int|None = None,
#         target_width: int|None = None,
#         offset_height: int|None = None,
#         offset_width: int|None = None,
#         mode: str|None = None,
#         image_proj: Tensor = None,
#         ip_scale: float =1.0,
#     ) -> tuple[Tensor, Tensor]:

#         ret_imgs, ret_txts, cache = [], [], {'offset_width': offset_width, 'txt_len': txt_len, 'h': current_height}

#         for i in range(0, current_width, offset_width):
#             start, end = i, min(i + target_width, current_width)
#             final_width = end - start

#             # shrink image, pe
#             width_idxs = torch.arange(start, end, dtype=torch.long)
#             small_img = self.shrink_img(img.clone(), current_height, current_width, width_idxs, final_width)
#             small_pe = self.shrink_pe(pe.clone(), txt_len, current_height, current_width, width_idxs, final_width)

#             # compute attention
#             cache |= {'i': i, 'w': final_width}
#             ret_img, ret_txt, cache = super().processor(self, small_img, txt, vec, small_pe, mode=mode, cache=cache)
#             ret_imgs.append(ret_img)
#             ret_txts.append(ret_txt)

#         return torch.cat(ret_imgs, 1), torch.cat(ret_txts, 1)


class IPSingleStreamBlockProcessor(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)

    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0
    ) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing
        ip_query = q
        ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_single_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)


        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")

        attn_out = attn_1 + ip_scale * ip_attention

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_out, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        return out


class SingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = output + self.proj_lora(torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weight
        output = x + mod.gate * output
        return output


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output

class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, x, vec, pe)
        else:
            return self.processor(self, x, vec, pe, image_proj, ip_scale)



class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

class ImageProjModel(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens





def extend_pe(curr_pe, prev_pe, txt_len, h, w, offset_width):
    # extract txt + img positional embeddings from curr_pe
    curr_txt_pe = curr_pe[:, :, :txt_len, :, :, :]  # (bs, 1, txt_len, pe_dim//2, 2, 2)
    curr_img_pe = curr_pe[:, :, txt_len:, :, :, :]  # (bs, 1, h_2*w_2, pe_dim//2, 2, 2)
    curr_img_pe = rearrange(curr_img_pe, "bs j (h w) pe_dim k l -> bs j pe_dim k l h w", h=h)
    curr_img_width = curr_img_pe.shape[-1]

    # extract img positional embeddings from prev_pe
    prev_img_pe = prev_pe[:, :, txt_len:, :, :, :]  # (bs, 1, h_2*w_2, pe_dim//2, 2, 2)
    prev_img_pe = rearrange(prev_img_pe, "bs j (h w) pe_dim k l -> bs j pe_dim k l h w", h=h)
    prev_img_width = prev_img_pe.shape[-1]

    # extend curr img pe with the offset_width first elements of prev img pe
    new_img_pe = torch.cat((prev_img_pe, curr_img_pe), dim=-1)
    idxs = torch.cat((torch.arange(offset_width), prev_img_width + torch.arange(curr_img_width)))
    new_img_pe = new_img_pe.index_select(-1, idxs)

    # reshape + put it back together
    new_img_pe = rearrange(new_img_pe, " bs j pe_dim k l h w -> bs j (h w) pe_dim k l")
    new_pe = torch.cat((curr_txt_pe, new_img_pe), dim=2)
    return new_pe


def test_extend_pe():

    bs, pe_dim, txt_len, h, w, offset_width, k, l = 1, 2, 1, 3, 3, 2, 1, 1
    num_elments = bs * (txt_len + h*w) * (pe_dim//2) * k * l
    curr_pe = torch.arange(0, num_elments).reshape(bs, 1, txt_len + h * w, pe_dim//2, k, l)
    prev_pe = torch.arange(-num_elments, 0).reshape(bs, 1, txt_len + h * w, pe_dim//2, k, l)

    new_pe = extend_pe(curr_pe, prev_pe, txt_len, h, w, offset_width)
    expected_new_pe = torch.tensor([[[[[[ 0]]],
                                 [[[-9]]],
                                 [[[-8]]],
                                 [[[ 1]]],
                                 [[[ 2]]],
                                 [[[ 3]]],
                                 [[[-6]]],
                                 [[[-5]]],
                                 [[[ 4]]],
                                 [[[ 5]]],
                                 [[[ 6]]],
                                 [[[-3]]],
                                 [[[-2]]],
                                 [[[ 7]]],
                                 [[[ 8]]],
                                 [[[ 9]]]]]])
    torch.testing.assert_close(new_pe, expected_new_pe)


def extend_img(curr_img, prev_img, h, w, offset_width):
    """extend img_k, img_q, or img_v"""
    # reshape imgs
    curr_img = rearrange(curr_img, "bs n_heads (h w) head_dim -> bs n_heads head_dim h w", h=h)
    prev_img = rearrange(prev_img, "bs n_heads (h w) head_dim -> bs n_heads head_dim h w", h=h)
    curr_img_width, prev_img_width = curr_img.shape[-1], prev_img.shape[-1]

    # extend curr img with the offset_width first elements of prev image
    new_img = torch.cat((prev_img, curr_img), dim=-1)
    idxs = torch.cat((torch.arange(offset_width), prev_img_width + torch.arange(curr_img_width)))
    new_img = new_img.index_select(-1, idxs)

    # reshape + put back together
    new_img = rearrange(new_img, "bs n_heads head_dim h w -> bs n_heads (h w) head_dim")
    return new_img

def test_extend_img():
    bs, num_heads, h, w, head_dim, offset_width = 1, 1, 2, 2, 1, 1
    num_elements = bs * num_heads * (h*w) * head_dim

    curr_img_q = torch.arange(0, num_elements).reshape(bs, num_heads, h*w, head_dim)
    prev_img_q = torch.arange(-num_elements, 0).reshape(bs, num_heads, h*w, head_dim)

    new_img_q = extend_img(curr_img_q, prev_img_q, h, w, offset_width)
    expected =  torch.tensor([[[[-4], [ 0], [ 1], [-2], [ 2], [ 3]]]])
    torch.testing.assert_close(new_img_q, expected)


def replace_img(curr_img, prev_img, h, w, offset_width):
    curr_img = rearrange(curr_img, "bs n_heads (h w) head_dim -> bs n_heads head_dim h w", h=h)
    prev_img = rearrange(prev_img, "bs n_heads (h w) head_dim -> bs n_heads head_dim h w", h=h)
    curr_img_width, prev_img_width = curr_img.shape[-1], prev_img.shape[-1]

    # get last offset_width entries from the previous image
    prev_idxs = prev_img_width-offset_width + torch.arange(offset_width)
    prev_img = prev_img.index_select(-1, prev_idxs)
    assert prev_img.shape[-1] == offset_width

    # skip the first offset_width entries from the current image
    curr_idxs = torch.arange(offset_width, curr_img_width)
    curr_img = curr_img.index_select(-1, curr_idxs)
    assert curr_img.shape[-1] == curr_img_width-offset_width

    # replace
    new_img = torch.cat((prev_img, curr_img), dim=-1)
    assert new_img.shape[-1] == curr_img_width

    # reshape + put back together
    new_img = rearrange(new_img, "bs n_heads head_dim h w -> bs n_heads (h w) head_dim")
    return new_img

def test_replace_img():
    bs, num_heads, h, w, head_dim = 1, 1, 6, 3, 1
    num_elements = bs * num_heads * (h*w) * head_dim

    offset_width = 1
    curr_img_q = torch.arange(0, num_elements).reshape(bs, num_heads, h*w, head_dim)
    prev_img_q = torch.arange(-num_elements, 0).reshape(bs, num_heads, h*w, head_dim)
    new_img_q = replace_img(curr_img_q, prev_img_q, h, w, offset_width)
    expected = torch.tensor([[[[-16], [  1], [  2], [-13], [  4], [  5], [-10], [  7], [  8], [ -7], [ 10], [ 11], [ -4], [ 13], [ 14], [ -1], [ 16], [ 17]]]])
    torch.testing.assert_close(new_img_q, expected)

    offset_width = 2
    curr_img_q = torch.arange(0, num_elements).reshape(bs, num_heads, h*w, head_dim)
    prev_img_q = torch.arange(-num_elements, 0).reshape(bs, num_heads, h*w, head_dim)
    new_img_q = replace_img(curr_img_q, prev_img_q, h, w, offset_width)
    expected =  torch.tensor([[[[-17], [-16], [  2], [-14], [-13], [  5], [-11], [-10], [  8], [ -8], [ -7], [ 11], [ -5], [ -4], [ 14], [ -2], [ -1], [ 17]]]])
    torch.testing.assert_close(new_img_q, expected)

class ReRoPEDoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, mode=None, cache=None, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # prepare pe
        pe_q, pe_k = pe, pe

        if mode is not None:
            if cache['i'] == 0:
                cache |=  {'img_k': img_k, 'img_v': img_v, 'pe':pe}
            else:
                if mode == 'extend':
                    # store for later use in the cache
                    img_k_tmp, img_v_tmp, pe_tmp = img_k, img_v, pe
                    # extend img_k, img_v, pe_k, pe_v
                    img_k = extend_img(img_k, cache['img_k'], cache['h'], cache['w'], cache['offset_width'])
                    img_v = extend_img(img_v, cache['img_v'], cache['h'], cache['w'], cache['offset_width'])
                    pe_k = extend_pe(pe, cache['pe'], cache['txt_len'], cache['h'], cache['w'], cache['offset_width'])
                    pe_q = pe
                    # update cache
                    cache |=  {'img_k': img_k_tmp, 'img_v': img_v_tmp, 'pe':pe_tmp}
                elif mode == 'replace':
                    # store for later use in the cache
                    img_k_tmp, img_v_tmp = img_k, img_v
                    img_k = replace_img(img_k, cache['img_k'], cache['h'], cache['w'], cache['offset_width'])
                    img_v = replace_img(img_v, cache['img_v'], cache['h'], cache['w'], cache['offset_width'])
                    # update cache
                    cache |=  {'img_k': img_k_tmp, 'img_v': img_v_tmp}
                else:
                    raise ValueError(f'mode should only equal replace or extend but got {mode=}')

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe_q, pe_k)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # cache |=  {'txt_k': txt_k, 'txt_v': txt_v, 'img_k': img_k, 'img_v': img_v, 'pe':pe}
        return img, txt, cache














class ReRoPEDoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = ReRoPEDoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def shrink_img(self, img, current_height, current_width, idxs, target_width):
        img = rearrange(img, "bs (h w) z -> bs z h w", h=current_height, w=current_width)
        img = torch.index_select(img, -1, idxs)
        img = rearrange(img, "bs z h w -> bs (h w) z", h=current_height, w=target_width)
        return img

    def shrink_pe(self, pe, txt_len, current_height, current_width, idxs, target_width):
        txt_pe = pe[:, :, :txt_len, :, :, :]  # (bs, 1, txt_len, pe_dim//2, 2, 2)
        img_pe = pe[:, :, txt_len:, :, :, :]  # (bs, 1, h_2*w_2, pe_dim//2, 2, 2)
        img_pe = rearrange(img_pe, "bs j (h w) pe_dim k l -> bs j pe_dim k l h w", h=current_height, w=current_width)
        img_pe = torch.index_select(img_pe, -1, idxs)
        img_pe = rearrange(img_pe, "bs j pe_dim k l h w ->bs j (h w) pe_dim k l", h=current_height, w=target_width)
        pe = torch.cat((txt_pe, img_pe), dim=2)
        return pe

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        txt_len: int,
        current_height: int,
        current_width: int,
        target_height: int|None = None,
        target_width: int|None = None,
        offset_height: int|None = None,
        offset_width: int|None = None,
        mode: str|None = None,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
    ) -> tuple[Tensor, Tensor]:
        # if image_proj is None:
        #     return self.processor(self, img, txt, vec, pe)
        # else:
        #     return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)

        ret_imgs, ret_txts, cache = [], [], {'offset_width': offset_width, 'txt_len': txt_len, 'h': current_height}

        for i in range(0, current_width, offset_width):
            start, end = i, min(i + target_width, current_width)
            final_width = end - start
            ic(i, final_width)

            # shrink image, pe
            width_idxs = torch.arange(start, end, dtype=torch.long, device=img.device)
            small_img = self.shrink_img(img.clone(), current_height, current_width, width_idxs, final_width)
            small_pe = self.shrink_pe(pe.clone(), txt_len, current_height, current_width, width_idxs, final_width)

            # compute attention
            cache |= {'i': i, 'w': final_width}
            ret_img, ret_txt, cache = self.processor(self, small_img, txt, vec, small_pe, mode=mode, cache=cache)
            ret_imgs.append(ret_img)
            ret_txts.append(ret_txt)

        ret_imgs, ret_txts = torch.cat(ret_imgs, 1), torch.cat(ret_txts, 1)
        ic(ret_imgs.shape, ret_txts.shape)
        return ret_imgs, ret_txts

