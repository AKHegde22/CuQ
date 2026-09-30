import torch
import torch.nn as nn
from tqdm import tqdm

def simulate_gptq_linear(linear_layer: nn.Linear, n_bits: int, act_hessian: torch.Tensor, q_group_size: int = 128):
    """
    Simulates GPTQ (Generative Pre-trained Transformer Quantization).
    Compensates for quantization error by distributing it to unquantized weights using the Inverse Hessian.
    For simulation speed and stability, we use a diagonal approximation of the Hessian (activations squared).
    """
    weight = linear_layer.weight.data.clone() # (out_features, in_features)
    if n_bits >= 16:
        return weight
        
    qmax = (1 << (n_bits - 1)) - 1
    in_features = weight.shape[1]
    
    # 1. Diagonal Hessian approximation + dampening
    H = act_hessian.clamp(min=1e-5)
    inv_H = 1.0 / H
    
    dequantized_weight = torch.zeros_like(weight)
    
    for i in range(0, in_features, q_group_size):
        end = min(i + q_group_size, in_features)
        block = weight[:, i:end]
        
        # Quantize the block
        block_max = block.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        block_scale = block_max / qmax
        
        q_block = torch.round(block / block_scale).clamp(-qmax, qmax)
        dq_block = q_block * block_scale
        
        dequantized_weight[:, i:end] = dq_block
        
        # Error compensation (GPTQ essence)
        # Error = Original - Quantized
        err = block - dq_block
        
        # Distribute error to remaining weights using inverse Hessian approximation
        if end < in_features:
            # We scale the error by the relative inverse Hessian of the remaining weights
            # This is a highly optimized diagonal simulation of the true dense GPTQ update
            remaining_inv_H = inv_H[end:]
            total_remaining = remaining_inv_H.sum()
            
            # Compensation factor
            comp = (err.sum(dim=-1, keepdim=True) * (remaining_inv_H.unsqueeze(0) / total_remaining))
            weight[:, end:] += comp
            
    return dequantized_weight

@torch.no_grad()
def apply_gptq_to_model(model, n_bits: int, calibration_dataloader, skip_layers=['unembed', 'embed'], q_group_size=128):
    """
    Applies simulated GPTQ to the HookedTransformer.
    """
    print(f"Applying Simulated GPTQ ({n_bits}-bit)...")
    
    # 1. Profile activations to get Hessian approximation
    act_sq = {}
    
    def get_act_hook(name):
        def hook(tensor, hook=None):
            # tensor is (batch, seq, in_features)
            # Diagonal Hessian ~ mean(x^2)
            sq = (tensor ** 2).mean(dim=(0, 1))
            if name not in act_sq:
                act_sq[name] = sq
            else:
                act_sq[name] += sq
        return hook
        
    hooks = []
    for i, block in enumerate(model.blocks):
        hooks.append((f"blocks.{i}.hook_mlp_in", get_act_hook(f"mlp_in_{i}")))
        hooks.append((f"blocks.{i}.hook_resid_pre", get_act_hook(f"attn_in_{i}")))
        
    with model.hooks(fwd_hooks=hooks):
        for batch in calibration_dataloader:
            model(batch)
            
    # Average the hessians over batches
    num_batches = len(calibration_dataloader)
    for k in act_sq:
        act_sq[k] /= num_batches
            
    modified_layers = 0
    for name, param in model.named_parameters():
        if any(skip in name for skip in skip_layers):
            continue
            
        if 'W_in' in name or 'W_out' in name or 'W_q' in name or 'W_k' in name or 'W_v' in name or 'W_o' in name:
            block_idx = int(name.split('.')[1])
            if 'mlp' in name:
                hessian = act_sq.get(f"mlp_in_{block_idx}")
            else:
                hessian = act_sq.get(f"attn_in_{block_idx}")
                
            if hessian is None or hessian.shape[0] != param.shape[-1]:
                hessian = torch.ones(param.shape[-1], device=param.device)
                
            param.data = simulate_gptq_linear(
                nn.Linear(param.shape[-1], param.shape[0], bias=False), 
                n_bits, 
                hessian, 
                q_group_size
            )
            modified_layers += 1
            
    print(f"Simulated GPTQ applied to {modified_layers} tensors.")
