import torch
from torch import nn, Tensor
from einops import rearrange, repeat
from icecream import install
install()

from flux.math import attention
from flux.modules.layers import Modulation, SelfAttention, EmbedND

class DoubleStreamBlockProcessor:
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


        if mode is not None and 'overlap' in cache and 'img_k' in cache and 'img_v' in cache:
            h = img.shape[-2]
            mask = torch.arange(h) < cache['overlap'] # 1's in overlap region
            # replace
            def replace(curr, prev, mask): return prev * mask[...::-1] + curr * ~mask
            def extend(curr, prev, mask): return torch.cat(prev[..., mask, :], curr)

            if mode == 'replace':
                img_k, img_v = replace(img_k, cache['img_k'], mask), replace(img_v, cache['img_v'], mask)
            elif mode == 'extend':
                img_k, img_v, pe = extend(img_k, cache['img_k'], mask), extend(img_v, cache['img_v'], mask), extend(pe, cache['pe'], mask)
            else:
                raise ValueError(f'mode should only equal replace or extend but got {mode=}')

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
        return img, txt, {'txt_k': txt_k, 'txt_v': txt_v, 'img_k': img_k, 'img_q': img_q, 'pe':pe}

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, mode: str = None):
        super().__init__()
        self.mode = mode
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
        # if image_proj is None:
        #     return self.processor(self, img, txt, vec, pe)
        # else:
        #     return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)

        if self.mode is None:
            return self.processor(self, img, txt, vec, pe, mode=None)

        h, HEIGHT, OVERLAP = img.shape[-2], 512, 256
        i, ret_imgs, ret_txts = 0, [], []
        cache = {'overlap': OVERLAP}

        if h > HEIGHT:
            ic(pe.shape)
            pe = pe[:, :HEIGHT]
            ic(pe.shape)

        while h > HEIGHT:
            ic(f'iteration={i}')
            small_img, img = img[..., :HEIGHT, :], img[..., OVERLAP:HEIGHT+OVERLAP]
            ic(small_img.shape)
            ret_img, ret_txt, cache = self.processor(self, small_img, txt, vec, pe, mode='replace', cache=cache)
            ret_imgs.append(ret_img)
            ret_txts.append(ret_txt)
            h = img.shape[-2:]
        return torch.cat(ret_imgs), torch.cat(ret_txts)

def main():
    # Hyperparameters
    hidden_size = 12
    num_heads = 2
    mlp_ratio = 2.0
    batch_size = 2
    img_seq_len = 768  # Total sequence length for image (to trigger chunking)
    txt_seq_len = 32
    pe_dim = hidden_size // num_heads
    axes_dim = [2,2,2]
    theta = 10_000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert sum(axes_dim) == pe_dim, f'{axes_dim=}\t{pe_dim=}'

    # Initialize model
    double_block = DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, mode=None).to(device)

    # Create dummy data
    img = torch.randn(batch_size, img_seq_len, hidden_size).to(device)  # [B, L_img, D]
    txt = torch.randn(batch_size, txt_seq_len, hidden_size).to(device)  # [B, L_txt, D]
    vec = torch.randn(batch_size, hidden_size).to(device)  # [B, D]

    # from sampling.py import prepare
    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    ids = torch.arange(img_seq_len + txt_seq_len, dtype=torch.float, device=device)
    ic(ids.shape)
    pe = EmbedND(pe_dim, theta=theta, axes_dim=axes_dim)(ids)
    # pe = pe.unsqueeze(0).unsqueeze(-1).expand(1, img_seq_len + txt_seq_len, hidden_size // num_heads)[..., None, None]  # [1, L_img + L_txt, D//H, 1, 1]
    ic(img.shape, txt.shape, vec.shape, pe.shape)

    # Run the model
    out_img, out_txt = double_block(img, txt, vec, pe)
    ic(out_img.shape, out_txt.shape)


if __name__ == "__main__":
    main()
