import math
import torch
from torch import nn, Tensor
from einops import rearrange, repeat
from icecream import install
install()

from flux.sampling import prepare, get_schedule, get_noise
from flux.util import configs
from flux.math import attention
from flux.modules.layers import Modulation, SelfAttention, EmbedND, MLPEmbedder, timestep_embedding

# **** replace, extend #####

def replace(curr, prev, mask, dims=None):
    """
    Wherever mask=1, replace that part of curr with the values of prev.
    But the values from prev are taken by flipping the mask along dims.

    Example:
    curr: tensor([0., 1., 2., 3., 4., 5., 6., 7.])
    prev: tensor([-7., -6., -5., -4., -3., -2., -1.,  0.])
    mask: tensor([ True,  True,  True,  True,  True,  True, False, False])
    replace(curr, prev, mask): tensor([ 0., -1., -2., -3., -4., -5.,  6.,  7.])
    """
    assert prev.shape == curr.shape == mask.shape, f'expected the same shapes but got {prev.shape=}\t{curr.shape=}\t{mask.shape=}'
    if dims is None: dims = tuple(range(len(curr.shape)))
    return (prev * mask.flip(dims)).flip(dims) + curr * ~mask

def extend(curr, prev, mask, dim=-1):
    """
    Concatenate prev to curr for all the values of prev where the mask is one.

    Example:
    curr: tensor([[0, 1, 2, 3],
                  [4, 5, 6, 7]])
    prev: tensor([[-7, -6, -5, -4],
                  [-3, -2, -1,  0]])
    mask: tensor([[ True,  True,  True,  True],
                  [False, False, False, False]])
    ret: tensor([[-7, -6, -5, -4],
                 [ 0,  1,  2,  3],
                 [ 4,  5,  6,  7]])
    """
    assert prev.shape == curr.shape == mask.shape, f'expected the same shapes but got {prev.shape=}\t{curr.shape=}\t{mask.shape=}'

    true_mask_shape = []
    for i in range(len(mask.shape)):
        s = mask.sum(i)
        if len(s.shape) == 0:
            true_dim = s.item()
        else:
            s_non_zero = s[s != 0] # zeros means the mask has no 1's in that row/col
            assert s_non_zero.unique().numel() == 1, f'mask is not rectangular on dim {i}\t{s_non_zero=}'
            true_dim = s_non_zero[0].item()
        true_mask_shape.append(true_dim)

    prev_reshaped = prev[mask].reshape(true_mask_shape)
    return torch.cat((prev_reshaped, curr), dim=dim)

def test_extend_1d():
    curr = torch.arange(0, 8)
    prev = torch.arange(-7, 1)
    mask = torch.arange(0, 8) < 6
    ret = extend(curr, prev, mask)
    assert torch.equal(ret, torch.Tensor([-7, -6, -5, -4, -3, -2,  0,  1,  2,  3,  4,  5,  6,  7]))

def test_extend_2d():
    curr = torch.arange(0, 8).reshape(2, 4)
    prev = torch.arange(-7, 1).reshape(2, 4)
    mask = (torch.arange(0, 8) < 4).reshape(2, 4)
    ret = extend(curr, prev, mask, dim=0)
    assert torch.equal(ret, torch.Tensor([[-7, -6, -5, -4], [ 0,  1,  2,  3], [ 4,  5,  6,  7]]))

def test_replace_1d():
    curr = torch.arange(0, 8)
    prev = torch.arange(-7, 1)
    mask = torch.arange(0, 8) < 6
    ret = replace(curr, prev, mask)
    assert torch.equal(ret, torch.Tensor([ 0., -1., -2., -3., -4., -5.,  6.,  7.]))

def test_replace_2d():
    curr = torch.arange(0, 8).reshape(2, 4)
    prev = torch.arange(-7, 1).reshape(2, 4)
    mask = (torch.arange(0, 8) < 6).reshape(2, 4)

    # one dimensional flip
    ret = replace(curr, prev, mask, dims=(1,))
    assert torch.equal(ret, torch.Tensor([[-4., -5., -6., -7.], [ 0., -1.,  6.,  7.]]))
    ret = replace(curr, prev, mask, dims=(0,))
    assert torch.equal(ret, torch.Tensor([[-3, -2, -1,  0], [-7, -6,  6,  7]]))

    # multi dimensional flip (this is what we want + should be correct)
    ret = replace(curr, prev, mask, dims=(0,1))
    assert torch.equal(ret, torch.Tensor([[ 0, -1, -2, -3], [-4, -5,  6,  7]]))
    ret = replace(curr, prev, mask)
    assert torch.equal(ret, torch.Tensor([[ 0, -1, -2, -3], [-4, -5,  6,  7]]))


# ***** Model ****

class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, mode=None, cache=None, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        ic(img.shape, txt.shape, vec.shape, pe.shape)
        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)
        ic(img_q.shape, img_k.shape, img_v.shape)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        if mode is not None:
            mask = torch.arange(img.shape[0]) < cache['overlap'] # 1's in overlap region

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

    def shrink_img(self, img, h, w, new_width, shift, i):
        idxs = torch.arange(new_width, dtype=torch.long)
        img = rearrange(img, "bs (h w) z -> bs z h w", h=h, w=w)
        img = torch.index_select(img, -1, idxs + i*shift)
        img = rearrange(img, "bs z h w -> bs (h w) z", h=h, w=new_width)
        return img

    def shrink_pe(self, pe, prompt_length, h, w, new_width, shift, i):
        idxs = torch.arange(new_width, dtype=torch.long)
        txt_pe = pe[:, :, :prompt_length, :, :, :]  # (bs, 1, prompt_length, pe_dim//2, 2, 2)
        img_pe = pe[:, :, prompt_length:, :, :, :]  # (bs, 1, h_2*w_2, pe_dim//2, 2, 2)
        img_pe = rearrange(img_pe, "bs j (h w) pe_dim k l -> bs j pe_dim k l h w", h=h, w=w)
        img_pe = torch.index_select(img_pe, -1, idxs + i*shift)
        img_pe = rearrange(img_pe, "bs j pe_dim k l h w ->bs j (h w) pe_dim k l", h=h, w=new_width)
        pe = torch.cat((txt_pe, img_pe), dim=2)
        return pe

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        mode: str|None = None,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
    ) -> tuple[Tensor, Tensor]:
        # if image_proj is None:
        #     return self.processor(self, img, txt, vec, pe)
        # else:
        #     return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)

        # height, width parameters
        height, width, ph, pw, prompt_length = 1024, 1024, 2, 2, 5
        h, w = 16 * (height // 16), 16 * (width // 16)
        h_1, w_1 = 2 * math.ceil(h / 16), 2 * math.ceil(w / 16)
        h_2, w_2 =  h_1//ph, w_1//pw

        # rerope parameters
        small_w, shift, i = 32, 16, 0
        ret_imgs, ret_txts = [], []
        cache = {'shift': shift}

        # make smaller image, pe
        small_img = self.shrink_img(img.clone(), h_2, w_2, small_w, shift, i)
        small_pe = self.shrink_pe(pe.clone(), prompt_length, h_2, w_2, small_w, shift, i)

        ret_img, ret_txt, cache = self.processor(self, small_img, txt, vec, small_pe, mode=mode, cache=cache)
        ret_imgs.append(ret_img)
        ret_txts.append(ret_txt)
        return torch.cat(ret_imgs), torch.cat(ret_txts)

def run():
    # init params
    seed, device = 42, 'cpu'
    torch.manual_seed(seed)
    flux_params = configs['flux-dev'].params
    width, height, num_steps = 1024, 1024, 1 # (1024, 1024, 25) are usd in main.py default params
    w, h = 16 * (width // 16), 16 * (height // 16) # round up to nearest multiple of 16, from XFluxPipeline.__call__
    bs, prompt_length, t5_hidden_size, clip_hidden_size = 1, 5, 4096, 768
    # ic(width, height, w, h)

    # init data
    # let h_1 = 2 * math.ceil(h / 16); w_1 = 2 * math.ceil(w / 16); c_img = 16
    original_img = get_noise(bs, h, w, device=device, dtype=torch.float, seed=seed) # (bs, c_img, h_1, w_1)
    prompt = torch.randn(bs, prompt_length)
    def t5(prompt):
        repetitions = (1,)*len(prompt.shape) + (t5_hidden_size,)
        return prompt.unsqueeze(-1).repeat(repetitions)
    def clip(prompt):
        prompt = prompt[:, 1] # clip ignores prompt_length which is along dim 1
        repetitions = (1,)*len(prompt.shape) + (clip_hidden_size,)
        return prompt.unsqueeze(-1).repeat(repetitions)
    inputs = prepare(t5, clip, img=original_img, prompt=prompt)

    # patch width = pw = 2; patch height = ph = 2; c_id = 3
    # let h_2 = h_1//ph, w_2 = w_1//pw
    img = inputs['img'] # (bs, (h_2 * w_2), (c_img * ph * pw))
    img_ids = inputs['img_ids'] # (bs, (h_2 * w_2), c_id)
    txt = inputs['txt'] # (b, prompt_length, t5_hidden_size)
    txt_ids = inputs['txt_ids'] # (bs, prompt_length, c_id)
    y = inputs['vec'] # (bs, clip_hidden_size)
    # ic(img.shape, img_ids.shape, txt.shape, txt_ids.shape, y.shape)

    # init model, based on model.py Flux.__init__()
    all_timesteps = get_schedule(num_steps, (w // 8) * (h // 8) // (16 * 16), shift=True)
    timesteps = torch.full((img.shape[0],), all_timesteps[0], dtype=img.dtype, device=img.device)
    pe_dim = flux_params.hidden_size // flux_params.num_heads
    img_in = nn.Linear(flux_params.in_channels, flux_params.hidden_size, bias=True)
    time_in = MLPEmbedder(in_dim=256, hidden_dim=flux_params.hidden_size)
    vector_in = MLPEmbedder(flux_params.vec_in_dim, flux_params.hidden_size)
    txt_in = nn.Linear(flux_params.context_in_dim, flux_params.hidden_size)
    pe_embedder = EmbedND(dim=pe_dim, theta=flux_params.theta, axes_dim=flux_params.axes_dim)
    block = DoubleStreamBlock(flux_params.hidden_size, flux_params.num_heads, flux_params.mlp_ratio)

    # run model, based on model.py Flux.forward()
    img = img_in(img)
    vec = time_in(timestep_embedding(timesteps, 256))
    vec = vec + vector_in(y)
    txt = txt_in(txt)
    ids = torch.cat((txt_ids, img_ids), dim=1) # (bs, h_2*w_2 + prompt_length, c)
    pe = pe_embedder(ids) # (bs, 1, h_2*w_2 + prompt_length, pe_dim//2, 2, 2) where the last 2,2 is due to RoPE's [[-sin(x),sin(x)],[-cos(x),cos(x)]]
    ic(img.shape, txt.shape, vec.shape, ids.shape, pe.shape)

    # the part we are modifying
    mode = None #'expand'
    out = block(img, txt, vec, pe, mode=mode)
    ic(out)


if __name__ == '__main__':
    run()
