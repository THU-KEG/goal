import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

# gated cross attention

class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # whether for text to only attend to immediate preceding image, or all images

        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(
        self,
        x,
        media,
        media_locations = None
    ):
        
        b, t, m = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')

        k, v = self.to_kv(media).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if exists(media_locations):
            text_time = media_locations.cumsum(dim = -1) # at each boolean of True, increment the time counter (relative to media time)
            media_time = torch.arange(t, device = x.device) + 1

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m = m))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
        media_locations = None  # boolean tensor indicating positions of media - (batch, sequence)
    ):
        x = self.attn(x, media, media_locations = media_locations) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh()  + x
        return x

if __name__ == '__main__':
    
    cross_attn = GatedCrossAttentionBlock(
        dim = 1024,
        dim_head = 64,
        heads = 8
    )
    
    text = torch.randn(1, 512, 1024)
    perceived = torch.randn(1, 2, 64, 1024)
    
    media_locations = torch.randint(0, 2, (1,512)).bool()
    
    text = cross_attn(
        text,
        perceived,
        media_locations = media_locations
    )


if __name__ == '__main__':
    
    cross_attn = GatedCrossAttentionBlock(
        dim = 1024,
        dim_head = 64,
        heads = 8
    )
    
    text = torch.randn(1, 512, 1024) # b, s, d
    perceived = torch.randn(1, 2, 64, 1024) # b, t, s, d
    
    media_locations = torch.randint(0, 2, (1,512)).bool()
    
    text = cross_attn(
        text,
        perceived,
        media_locations = media_locations
    )

    import ipdb
    ipdb.set_trace()