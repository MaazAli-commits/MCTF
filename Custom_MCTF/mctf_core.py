import math
import torch
from typing import Callable, Tuple, Dict, Any

def mctf(
    x: torch.Tensor,
    attn: torch.Tensor,
    size: torch.Tensor,
    r: int,
    num_protected: int = 1,
    tau_sim: float = 16.0,
    tau_info: float = 1.0,
    tau_size: float = 40.0,
    bidirection: bool = False,
    one_step_ahead: bool = False,
    pooling_type: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-criteria Token Fusion (MCTF).
    Args:
        x: Token embeddings, shape (B, T, D)
        attn: Attention map from the current layer, shape (B, H, T, T)
        size: Token sizes (number of raw tokens merged into this token), shape (B, T, 1)
        r: Number of tokens to reduce in this block.
        num_protected: Number of tokens to protect from merging (usually 1 for CLS).
        tau_sim, tau_info, tau_size: Temperatures for the three criteria.
        bidirection: If True, uses bidirectional bipartite matching.
        one_step_ahead: If True, calculates the approximate attention for the next block.
        pooling_type: Strategy for merging token sizes (0 is standard sum).
    Returns:
        x_merged: (B, T - r, D)
        size_merged: (B, T - r, 1)
        attn_n: The one-step-ahead attention matrix (or None), shape (B, H, T-r, T-r) if one_step_ahead=True
    """
    B, T, D = x.shape
    
    # Cap r to maximum 50% reduction of non-protected tokens
    if T - r < 10: 
        r = T - 10
    max_r = (T - num_protected) // 2
    if r > max_r:
        r = max_r
        
    if r <= 0:
        return x, size, (attn if one_step_ahead else None)

    # 1. Calculate the bipartite matching indices and get the merge closure
    merge_fn = bipartite_soft_matching(
        metric=x,
        attn=attn,
        size=size,
        r=r,
        num_protected=num_protected,
        tau_sim=tau_sim,
        tau_info=tau_info,
        tau_size=tau_size,
        bidirection=bidirection
    )

    # 2. Perform the weighted average token fusion
    x_merged, size_merged, attn_n = merge_wavg(
        merge=merge_fn,
        x=x,
        size=size,
        attn=attn,
        one_step_ahead=one_step_ahead,
        pooling_type=pooling_type
    )

    return x_merged, size_merged, attn_n


def bipartite_soft_matching(
    metric: torch.Tensor,
    attn: torch.Tensor,
    size: torch.Tensor,
    r: int,
    num_protected: int,
    tau_sim: float,
    tau_info: float,
    tau_size: float,
    bidirection: bool
) -> Callable:
    """
    Finds the bipartite matching between tokens and returns a `merge` closure function
    that applies the reduction to any tensor.
    """
    B, T, _ = metric.shape
    
    if bidirection:
        r1, r2 = r // 2, r - (r // 2)
    else:
        r1, r2 = r, 0

    with torch.no_grad():
        # L2 Normalize the metric (features)
        metric = metric / metric.norm(dim=-1, keepdim=True)
        # Split tokens into two sets (even and odd indices) for bipartite matching
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        # 1. Similarity Criterion
        if tau_sim > 0:
            W_sim = a @ b.transpose(-1, -2)
            W_sim = ((W_sim + 1) / 2) ** (1 / tau_sim)
        else:
            W_sim = torch.ones((a.shape[0], a.shape[1], b.shape[1]), device=a.device)

        # 2. Informativeness Criterion
        if tau_info > 0 and attn is not None:
            # attn shape: (B, H, T, T). We average over heads and queries (dim 1 and 2)
            # to get the importance of each key (token).
            attn_mean = 1 / (attn.mean(dim=[1, 2]) + 1e-8)
            attn_mean = attn_mean / attn_mean.max(dim=1, keepdim=True)[0]
            attn_a = attn_mean[..., ::2, None]
            attn_b = attn_mean[..., 1::2, None].transpose(1, 2)
            W_info = (attn_a * attn_b) ** (1 / tau_info)
        else:
            W_info = 1.0

        # 3. Token-size Criterion
        if tau_size > 0 and size is not None:
            size_inv = 1 / (size + 1e-8)
            size_inv = size_inv / size_inv.max(dim=1, keepdim=True)[0]
            size_a = size_inv[..., ::2, :]
            size_b = size_inv[..., 1::2, :].transpose(1, 2)
            W_size = (size_a * size_b) ** (1 / tau_size)
        else:
            W_size = 1.0

        # Final joint scores
        scores = W_sim * W_info * W_size
        
        # Protect CLS token (and any other protected tokens at the start)
        if num_protected > 0:
            scores[..., :num_protected, :] = -math.inf
            
        n, t1, t2 = scores.shape

        # Find best matches for set A (even tokens)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r1:, :]  # Unmerged Tokens in A
        src_idx = edge_idx[..., :r1, :]  # Merged Tokens in A (Source)
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # Their targets in B (Destination)

        if num_protected > 0:
            unm_idx = unm_idx.sort(dim=1)[0]

        # Bidirectional matching (matching from B to A for the remaining reduction r2)
        if bidirection and r2 > 0:
            new_scores = scores.gather(dim=-2, index=unm_idx.repeat(1, 1, t2)).transpose(1, 2)
            node_max2, node_idx2 = new_scores.max(dim=-1)
            edge_max2, edge_idx2 = node_max2.sort(dim=-1, descending=True)
            edge_idx2 = edge_idx2[..., None]
            unm_idx2 = edge_idx2[..., r2:, :]
            src_idx2 = edge_idx2[..., :r2, :]
            dst_idx2 = node_idx2[..., None].gather(dim=-2, index=src_idx2)
        else:
            unm_idx2, src_idx2, dst_idx2 = None, None, None

    def dim_match(src, dim=1, dim_num=5):
        while len(src.shape) < dim_num:
            src = src.unsqueeze(dim)
        return src

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        """
        Closure that applies the computed bipartite matching indices to reduce tensor x.
        x usually has shape (B, T, D) or (B, H, T_Q, T_K).
        """
        src = x[..., ::2, :]
        dst = x[..., 1::2, :]

        b, mid, c = src.shape[0], src.shape[1:-2], src.shape[-1]
        dim_num = len(src.shape)
        
        unm_idx_ = dim_match(unm_idx, dim=1, dim_num=dim_num)
        src_idx_ = dim_match(src_idx, dim=1, dim_num=dim_num)
        dst_idx_ = dim_match(dst_idx, dim=1, dim_num=dim_num)
        
        unm = src.gather(dim=-2, index=unm_idx_.expand(b, *mid, t1 - r1, c))
        src_subset = src.gather(dim=-2, index=src_idx_.expand(b, *mid, r1, c))
        dst = dst.scatter_reduce(-2, dst_idx_.expand(b, *mid, r1, c), src_subset, reduce=mode)
        
        if bidirection and r2 > 0:
            unm_idx2_ = dim_match(unm_idx2, dim=1, dim_num=dim_num)
            src_idx2_ = dim_match(src_idx2, dim=1, dim_num=dim_num)
            dst_idx2_ = dim_match(dst_idx2, dim=1, dim_num=dim_num)

            src2, dst2 = dst, unm
            unm2 = src2.gather(dim=-2, index=unm_idx2_.expand(b, *mid, t2 - r2, c))
            src2_subset = src2.gather(dim=-2, index=src_idx2_.expand(b, *mid, r2, c))
            dst2 = dst2.scatter_reduce(-2, dst_idx2_.expand(b, *mid, r2, c), src2_subset, reduce=mode)
            out = torch.cat([dst2, unm2], dim=-2)
        else:
            out = torch.cat([unm, dst], dim=-2)
            
        return out

    return merge


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: torch.Tensor,
    attn: torch.Tensor = None,
    one_step_ahead: bool = False,
    pooling_type: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    """
    attn_n = None
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    if pooling_type == 0:
        # Standard Weighted Average by token size
        x_merged = merge(x * size, mode="sum")
        
        if one_step_ahead and attn is not None:
            # attn shape: (B, H, T_Q, T_K) -> Merge Queries
            attn_n = merge(attn[..., None], mode="sum").squeeze(-1)
            # Merge Keys weighted by size
            attn_n = merge(attn_n * size[:, None], mode="sum").squeeze(-1)
            
        with torch.no_grad():
            size_merged = merge(size, mode="sum")
            
        if one_step_ahead and attn is not None:
            attn_n = attn_n / (size_merged[:, None] + 1e-8)
            
        x_merged = x_merged / (size_merged + 1e-8)
        
    else:
        # Fallback for other pooling types if needed (simplified to MEAN here)
        x_merged = merge(x, mode="mean")
        with torch.no_grad():
            size_merged = merge(size, mode="sum")
        if one_step_ahead and attn is not None:
            attn_n = merge(attn[..., None], mode="mean").squeeze(-1)
            attn_n = merge(attn_n, mode="mean").squeeze(-1)

    return x_merged, size_merged, attn_n
