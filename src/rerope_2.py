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


def extend(curr, prev, mask, dim=2):
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


# **** new expand ****

def expand_pe(curr_pe, prev_pe, txt_len, h, w, offset_width):
    # extract txt + img positional embeddings from curr_pe
    curr_txt_pe = curr_pe[:, :, :txt_len, :, :, :]  # (bs, 1, txt_len, pe_dim//2, 2, 2)
    curr_img_pe = curr_pe[:, :, txt_len:, :, :, :]  # (bs, 1, h_2*w_2, pe_dim//2, 2, 2)
    curr_img_pe = rearrange(curr_img_pe, "bs j (h w) pe_dim k l -> bs j pe_dim k l h w", h=h)
    curr_img_width = w

    # extract img positional embeddings from prev_pe
    prev_img_pe = prev_pe[:, :, txt_len:, :, :, :]  # (bs, 1, h_2*w_2, pe_dim//2, 2, 2)
    prev_img_pe = rearrange(prev_img_pe, "bs j (h w) pe_dim k l -> bs j pe_dim k l h w", h=h)
    prev_img_width = w

    # expand curr img pe with the offset_width first elements of prev img pe
    new_img_pe = torch.cat((prev_img_pe, curr_img_pe), dim=-1)
    idxs = torch.cat((torch.arange(offset_width), prev_img_width + torch.arange(curr_img_width)))
    new_img_pe = new_img_pe.index_select(-1, idxs)

    # reshape + put it back together
    new_img_pe = rearrange(new_img_pe, " bs j pe_dim k l h w -> bs j (h w) pe_dim k l")
    new_pe = torch.cat((curr_txt_pe, new_img_pe), dim=2)
    return new_pe


def test_expand_pe():

    bs, pe_dim, txt_len, h, w, offset_width, k, l = 1, 2, 1, 3, 3, 2, 1, 1
    num_elments = bs * (txt_len + h*w) * (pe_dim//2) * k * l
    curr_pe = torch.arange(0, num_elments).reshape(bs, 1, txt_len + h * w, pe_dim//2, k, l)
    prev_pe = torch.arange(-num_elments, 0).reshape(bs, 1, txt_len + h * w, pe_dim//2, k, l)

    new_pe = expand_pe(curr_pe, prev_pe, txt_len, h, w, offset_width)
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
    """Expand img_k, img_q, or img_v"""
    # reshape imgs
    curr_img = rearrange(curr_img, "bs n_heads (h w) head_dim -> bs n_heads head_dim h w", h=h)
    prev_img = rearrange(prev_img, "bs n_heads (h w) head_dim -> bs n_heads head_dim h w", h=h)
    curr_img_width, prev_img_width = curr_img.shape[-1], prev_img.shape[-1]

    # expand curr img with the offset_width first elements of prev image
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
                    pe_k = expand_pe(pe, cache['pe'], cache['txt_len'], cache['h'], cache['w'], cache['offset_width'])
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

            # shrink image, pe
            width_idxs = torch.arange(start, end, dtype=torch.long)
            small_img = self.shrink_img(img.clone(), current_height, current_width, width_idxs, final_width)
            small_pe = self.shrink_pe(pe.clone(), txt_len, current_height, current_width, width_idxs, final_width)
            ic(i, start, end)

            # compute attention
            cache |= {'i': i, 'w': final_width}
            ret_img, ret_txt, cache = self.processor(self, small_img, txt, vec, small_pe, mode=mode, cache=cache)
            ret_imgs.append(ret_img)
            ret_txts.append(ret_txt)

        return torch.cat(ret_imgs, 1), torch.cat(ret_txts, 1)

def run():
    # init params
    seed, device = 42, 'cpu'
    torch.manual_seed(seed)
    flux_params = configs['flux-dev'].params
    bs, txt_len, t5_hidden_size, clip_hidden_size, num_steps = 1, 5, 4096, 768, 1

    # width, height: the user specified width and height
    # w, h: the width, height rounded up to nearest multiple of 16; this becomes the effective input size
    # current_width, current_height: after several layers/computations/reshaping the image gets shrunk to these dimensions; most calculations use this image size
    width, height, ph, pw = 1024, 1024, 2, 2 # from main.py default params
    w, h = 16 * (width // 16), 16 * (height // 16) # round up to nearest multiple of 16, from XFluxPipeline.__call__
    w_1, h_1 = 2 * math.ceil(w / 16), 2 * math.ceil(h / 16)
    current_width, current_height = w_1//pw, h_1//ph

    # init data | c_img = 16,  c_id = 3
    def t5(prompt):
        repetitions = (1,)*len(prompt.shape) + (t5_hidden_size,)
        return prompt.unsqueeze(-1).repeat(repetitions)
    def clip(prompt):
        prompt = prompt[:, 1] # clip ignores txt_len which is along dim 1
        repetitions = (1,)*len(prompt.shape) + (clip_hidden_size,)
        return prompt.unsqueeze(-1).repeat(repetitions)
    prompt = torch.randn(bs, txt_len) # (bs, txt_len)
    original_img = get_noise(bs, h, w, device=device, dtype=torch.float, seed=seed) # (bs, c_img, h_1, w_1)
    inputs = prepare(t5, clip, img=original_img, prompt=prompt)
    img = inputs['img'] # (bs, (h_2 * w_2), (c_img * ph * pw))
    img_ids = inputs['img_ids'] # (bs, (h_2 * w_2), c_id)
    txt = inputs['txt'] # (b, txt_len, t5_hidden_size)
    txt_ids = inputs['txt_ids'] # (bs, txt_len, c_id)
    y = inputs['vec'] # (bs, clip_hidden_size)

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
    ids = torch.cat((txt_ids, img_ids), dim=1) # (bs, h_2*w_2 + txt_len, c)
    pe = pe_embedder(ids) # (bs, 1, h_2*w_2 + txt_len, pe_dim//2, 2, 2) where the last 2,2 is due to RoPE's [[-sin(x),sin(x)],[-cos(x),cos(x)]]

    # ReRoPE
    mode = 'replace' # 'extend'
    target_width, offset_width = 32, 16
    out = block(
        img, txt, vec, pe, txt_len=txt_len,
        current_width=current_width, current_height=current_height,
        target_width=target_width, offset_width=offset_width,
        mode=mode,
        )
    ic(out[0].shape, out[1].shape)


if __name__ == '__main__':
    test_replace_img()
    test_expand_pe()
    test_extend_img()
    run()
