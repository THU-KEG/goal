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

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents, att_mask=None):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        # b, t, n = att_mask.shape
        b, t, n ,d = x.shape
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention
        import ipdb
        ipdb.set_trace()
        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        if att_mask is not None:
            att_mask = torch.cat([att_mask, torch.ones(b, t, latents.shape[2])], dim=2)
            att_mask = repeat(att_mask, 'b t n -> b h t n', h=h)
            att_weights = torch.where(att_mask>0, 0.0, torch.finfo(sim.dtype).min)
            att_weights = att_weights.unsqueeze(3)
            sim += att_weights

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_time_embeds = 4,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, att_mask=None):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')
        if att_mask and att_mask.ndim == 2:
            att_mask = rearrange(att_mask, 'b n -> b 1 n')

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents, att_mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)

    
if __name__ == '__main__':

    perceive = PerceiverResampler(
        dim = 1024,
        depth = 2,
        dim_head = 64,
        heads = 8,
        num_latents = 64,    # the number of latents to shrink your media sequence to, perceiver style
        num_time_embeds = 4  # say you have 4 images maximum in your dialogue
    )

    medias = torch.randn(1, 2, 256, 1024) # (batch, time, sequence length, dimension)
    att_mask = torch.ones(1, 2, 256)
    att_mask[:, :, -1] = 0
    perceived = perceive(medias, att_mask=att_mask) # (1, 2, 64, 1024) - (batch, time, num latents, dimension)

    import ipdb
    ipdb.set_trace()
