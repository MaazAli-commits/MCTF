import torch
import torch.nn as nn
import timm
from mctf_core import mctf

class MCTFVisionTransformer(nn.Module):
    """
    A clean wrapper that injects Multi-criteria Token Fusion (MCTF) into a standard Vision Transformer.
    We use forward hooks to grab the attention maps without having to rewrite the entire ViT architecture!
    """
    def __init__(
        self, 
        model_name='deit_tiny_patch16_224', 
        pretrained=True, 
        num_classes=100,
        r_schedule=None,  # e.g. [0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16, 0]
        tau_sim=16.0,
        tau_info=1.0,
        tau_size=40.0,
        bidirection=True
    ):
        super().__init__()
        
        # 1. Load a standard baseline ViT from timm
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # 2. MCTF Hyperparameters
        self.num_layers = len(self.vit.blocks)
        if r_schedule is None:
            # Default: reduce 10 tokens per layer after the first 3 layers
            self.r_schedule = [0]*3 + [10]*(self.num_layers - 3)
        else:
            self.r_schedule = r_schedule
            
        self.tau_sim = tau_sim
        self.tau_info = tau_info
        self.tau_size = tau_size
        self.bidirection = bidirection



    def forward_features(self, x):
        """
        Custom forward pass for features that interleaves the ViT blocks with MCTF reductions.
        """
        # Standard ViT embedding steps
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        
        # Handle distillation token if present, otherwise just CLS token
        if hasattr(self.vit, 'dist_token') and self.vit.dist_token is not None:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            num_protected = 2
        else:
            x = torch.cat((cls_token, x), dim=1)
            num_protected = 1
            
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        # Initialize token sizes for MCTF bookkeeping (all tokens start as size 1)
        B, T, D = x.shape
        size = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        
        # Pass through the Transformer blocks sequentially
        for i, block in enumerate(self.vit.blocks):
            r = self.r_schedule[i]
            
            if r > 0:
                # 1. Compute A^(l+1) on the full X^l (Look one step ahead)
                with torch.no_grad():
                    B_n, N_n, C_n = x.shape
                    x_norm = block.norm1(x)
                    qkv = block.attn.qkv(x_norm).reshape(B_n, N_n, 3, block.attn.num_heads, C_n // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                    q, k = qkv[0], qkv[1]
                    A_next = (q @ k.transpose(-2, -1)) * block.attn.scale
                    
                    # Handle Size-aware attention if configured (for precise mode)
                    if self.tau_size > 0 and size is not None:
                        A_next = A_next + size.log()[:, None, None, :, 0]
                        
                    A_next = A_next.softmax(dim=-1)

                # Are we doing Approximate (1) or Precise (2)?
                # Precise mode (one_step_ahead=2) recomputes attention on reduced tokens.
                # Approximate mode (one_step_ahead=1) aggregates A_next -> A_n and reuses it.
                precise_mode = False # Set to False to match the paper's main FLOP-saving mode

                if precise_mode:
                    # Precise Mode: Fuse X^l, then run the block normally to compute a fresh Q@K
                    x, size, _ = mctf(
                        x=x, attn=A_next, size=size, r=r,
                        num_protected=num_protected, tau_sim=self.tau_sim, 
                        tau_info=self.tau_info, tau_size=self.tau_size,
                        bidirection=self.bidirection, one_step_ahead=False
                    )
                    x = block(x)
                else:
                    # Approximate Mode: Fuse X^l and aggregate A_next -> A_n
                    x, size, A_n = mctf(
                        x=x, attn=A_next, size=size, r=r,
                        num_protected=num_protected, tau_sim=self.tau_sim, 
                        tau_info=self.tau_info, tau_size=self.tau_size,
                        bidirection=self.bidirection, one_step_ahead=True
                    )
                    
                    # 3. Monkey-patch block.attn to bypass Q@K computation and use A_n!
                    original_attn_forward = block.attn.forward
                    
                    def fused_attn_forward(x_in, *args, **kwargs):
                        B_, N_, C_ = x_in.shape
                        qkv_ = block.attn.qkv(x_in).reshape(B_, N_, 3, block.attn.num_heads, C_ // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                        v_ = qkv_[2]
                        
                        attn_ = block.attn.attn_drop(A_n) # Use our aggregated A_n
                        x_out = (attn_ @ v_).transpose(1, 2).reshape(B_, N_, C_)
                        return block.attn.proj_drop(block.attn.proj(x_out))
                    
                    # Temporarily replace the forward method
                    block.attn.forward = fused_attn_forward
                    
                    # Run the block for real on the reduced tokens (will only compute V and MLP)
                    x = block(x)
                    
                    # Restore original forward
                    block.attn.forward = original_attn_forward
            else:
                # No reduction, standard forward pass
                x = block(x)
                
        x = self.vit.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # Use only the CLS token for classification
        if hasattr(self.vit, 'head'):
            x = self.vit.head(x[:, 0])
        elif hasattr(self.vit, 'forward_head'):
            x = self.vit.forward_head(x[:, 0])
        return x

def create_baseline_vit(model_name='deit_tiny_patch16_224', pretrained=True, num_classes=100):
    """Helper to create a standard non-MCTF ViT for baseline comparison"""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

# Quick sanity check test
if __name__ == "__main__":
    model = MCTFVisionTransformer(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (CLS logits): {out.shape}")
    print("MCTF-ViT forward pass successful!")
