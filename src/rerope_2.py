import torch
from torch import nn, Tensor
from einops import rearrange, repeat
from icecream import install

from flux.sampling import prepare
install()

from flux.math import attention
from flux.modules.layers import Modulation, SelfAttention, EmbedND

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

        if mode is not None:
            h = img.shape[-2]
            mask = torch.arange(h) < cache['overlap'] # 1's in overlap region

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

        h, HEIGHT, OVERLAP = img.shape[-1], 512, 256
        ret_imgs, ret_txts = [], []
        cache = {'overlap': OVERLAP}

        while h > HEIGHT:
            rearrange(img, "b (h w) c -> b h w c", h=HEIGHT)
            idxs = torch.arange(HEIGHT, dtype=torch.long)
            torch.index_select(img, 1, idxs)
            small_img, img = img[..., :HEIGHT, :], img[..., OVERLAP:HEIGHT+OVERLAP]
            ret_img, ret_txt, cache = self.processor(self, small_img, txt, vec, pe, mode=mode, cache=cache)
            ret_imgs.append(ret_img)
            ret_txts.append(ret_txt)
            h = img.shape[-2:]
        return torch.cat(ret_imgs), torch.cat(ret_txts)

def run():
    torch.manual_seed(42)
    # from flux-dev ModelSpec
    hidden_size, num_heads, mlp_ratio = 3072, 24, 4.0

    # bs, c, h, w, hidden_text_size = 1, 3, 16, 16, 16
    # image = torch.randn(bs, c, h, w)
    # prompt = torch.randn(bs, hidden_text_size)
    # prep_inputs = prepare(lambda x: x, lambda x: x, image, prompt)
    # img, img_ids, txt, txt_ids, vec = prep_inputs.values()
    # ic(img.shape, img_ids.shape, txt.shape, txt_ids.shape, vec.shape)
    img = torch.randn(1, 4096, 3072)
    txt = torch.randn(1, 512, 3072)
    vec = torch.randn(1, 3072)
    pe = torch.randn(1, 1, 4608, 64, 2, 2)

    block = DoubleStreamBlock(hidden_size, num_heads, mlp_ratio)
    out = block(img, txt, vec, pe)
    ic(out)



if __name__ == '__main__':
    run()
