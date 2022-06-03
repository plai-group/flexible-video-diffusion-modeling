import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from .nn import zero_module, normalization, checkpoint


class RPENet(nn.Module):
    def __init__(self, channels, num_heads, time_embed_dim):
        super().__init__()
        self.embed_distances = nn.Linear(3, channels)
        self.embed_diffusion_time = nn.Linear(time_embed_dim, channels)
        self.silu = nn.SiLU()
        self.out = nn.Linear(channels, channels)
        self.out.weight.data *= 0.
        self.out.bias.data *= 0.
        self.channels = channels
        self.num_heads = num_heads

    def forward(self, temb, relative_distances):
        distance_embs = th.stack(
            [th.log(1+(relative_distances).clamp(min=0)),
             th.log(1+(-relative_distances).clamp(min=0)),
             (relative_distances == 0).float()],
            dim=-1
        )  # BxTxTx3
        B, T, _ = relative_distances.shape
        C = self.channels
        emb = self.embed_diffusion_time(temb).view(B, T, 1, C) \
            + self.embed_distances(distance_embs)  # B x T x T x C
        return self.out(self.silu(emb)).view(*relative_distances.shape, self.num_heads, self.channels//self.num_heads)


class RPE(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py
    def __init__(self, channels, num_heads, time_embed_dim, use_rpe_net=False):
        """ This module handles the relative positional encoding.
        Args:
            channels (int): Number of input channels.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.use_rpe_net = use_rpe_net
        if use_rpe_net:
            self.rpe_net = RPENet(channels, num_heads, time_embed_dim)
        else:
            self.lookup_table_weight = nn.Parameter(
                th.zeros(2 * self.beta + 1,
                         self.num_heads,
                         self.head_dim))

    def get_R(self, pairwise_distances, temb):
        if self.use_rpe_net:
            return self.rpe_net(temb, pairwise_distances)
        else:
            return self.lookup_table_weight[pairwise_distances]  # BxTxTxHx(C/H)

    def forward(self, x, pairwise_distances, temb, mode):
        if mode == "qk":
            return self.forward_qk(x, pairwise_distances, temb)
        elif mode == "v":
            return self.forward_v(x, pairwise_distances, temb)
        else:
            raise ValueError(f"Unexpected RPE attention mode: {mode}")

    def forward_qk(self, qk, pairwise_distances, temb):
        # qv is either of q or k and has shape BxDxHxTx(C/H)
        # Output shape should be # BxDxHxTxT
        R = self.get_R(pairwise_distances, temb)
        return th.einsum(  # See Eq. 16 in https://arxiv.org/pdf/2107.14222.pdf
            "bdhtf,btshf->bdhts", qk, R  # BxDxHxTxT
        )

    def forward_v(self, attn, pairwise_distances, temb):
        # attn has shape BxDxHxTxT
        # Output shape should be # BxDxHxYx(C/H)
        R = self.get_R(pairwise_distances, temb)
        th.einsum("bdhts,btshf->bdhtf", attn, R)
        return th.einsum(  # See Eq. 16ish in https://arxiv.org/pdf/2107.14222.pdf
            "bdhts,btshf->bdhtf", attn, R  # BxDxHxTxT
        )

    def forward_safe_qk(self, x, pairwise_distances, temb):
        R = self.get_R(pairwise_distances, temb)
        B, T, _, H, F = R.shape
        D = x.shape[1]
        res = x.new_zeros(B, D, H, T, T) # attn shape
        for b in range(B):
            for d in range(D):
                for h in range(H):
                    for i in range(T):
                        for j in range(T):
                            res[b, d, h, i, j] = x[b, d, h, i].dot(R[b, i, j, h])
        return res


class RPEAttention(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py#L42
    def __init__(self, channels, num_heads, use_checkpoint=False,
                 time_embed_dim=None, use_rpe_net=None,
                 use_rpe_q=True, use_rpe_k=True, use_rpe_v=True,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = zero_module(nn.Linear(channels, channels))
        self.norm = normalization(channels)

        if use_rpe_q or use_rpe_k or use_rpe_v:
            assert use_rpe_net is not None
        def make_rpe_func():
            return RPE(
                channels=channels, num_heads=num_heads,
                time_embed_dim=time_embed_dim, use_rpe_net=use_rpe_net,
            )
        self.rpe_q = make_rpe_func() if use_rpe_q else None
        self.rpe_k = make_rpe_func() if use_rpe_k else None
        self.rpe_v = make_rpe_func() if use_rpe_v else None

    def forward(self, x, temb, frame_indices, attn_mask=None, attn_weights_list=None):
        out, attn = checkpoint(self._forward, (x, temb, frame_indices, attn_mask), self.parameters(), self.use_checkpoint)
        if attn_weights_list is not None:
            B, D, C, T = x.shape
            attn_weights_list.append(attn.detach().view(B*D, -1, T, T).mean(dim=1).abs())  # logging attn weights
        return out

    def _forward(self, x, temb, frame_indices, attn_mask):
        B, D, C, T = x.shape
        x = x.reshape(B*D, C, T)
        x = self.norm(x)
        x = x.view(B, D, C, T)
        x = th.einsum("BDCT -> BDTC", x)  # just a permutation
        qkv = self.qkv(x).reshape(B, D, T, 3, self.num_heads, C // self.num_heads)
        qkv = th.einsum("BDTtHF -> tBDHTF", qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # q, k, v shapes: BxDxHxTx(C/H)
        q *= self.scale
        attn = (q @ k.transpose(-2, -1)) # BxDxHxTxT
        if self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None:
            pairwise_distances = (frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(-2)) # BxTxT
        # relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pairwise_distances, temb=temb, mode="qk")
        # relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale, pairwise_distances, temb=temb, mode="qk").transpose(-1, -2)

        # softmax where all elements with mask==0 can attend to eachother and all with mask==1
        # can attend to eachother (but elements with mask==0 can't attend to elements with mask==1)
        def softmax(w, attn_mask):
            if attn_mask is not None:
                allowed_interactions = attn_mask.view(B, 1, T) * attn_mask.view(B, T, 1)
                allowed_interactions += (1-attn_mask.view(B, 1, T)) * (1-attn_mask.view(B, T, 1))
                inf_mask = (1-allowed_interactions)
                inf_mask[inf_mask == 1] = th.inf
                w = w - inf_mask.view(B, 1, 1, T, T)  # BxDxHxTxT
            return th.softmax(w.float(), dim=-1).type(w.dtype)

        attn = softmax(attn, attn_mask)
        out = attn @ v
        # relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn, pairwise_distances, temb=temb, mode="v")
        out = th.einsum("BDHTF -> BDTHF", out).reshape(B, D, T, C)
        out = self.proj_out(out)
        x = x + out
        x = th.einsum("BDTC -> BDCT", x)
        return x, attn