import torch
import torch.nn as nn
from tqdm import tqdm

def simulate_awq_linear(linear_layer: nn.Linear, n_bits: int, act_scales: torch.Tensor, q_group_size: int = 128):
    """
    Simulates Activation-aware Weight Quantization (AWQ).
    AWQ scales up weights corresponding to salient activation channels to reduce their quantization error.
    """
    weight = linear_layer.weight.data.clone() # (out_features, in_features)
    if n_bits >= 16:
        return weight
        
    qmax = (1 << (n_bits - 1)) - 1
    
    # 1. Determine scaling factor based on activation scales
    # Simplified AWQ scaling: we scale weights by a factor of sqrt(act_scales) to protect salient channels.
    # In full AWQ, this is found via grid search, but this is a solid heuristic.
    alpha = 0.5
    scales = (act_scales ** alpha).clamp(min=1e-4)
    scales = scales / scales.max()
    
    # Apply scales to weights (simulating the protection)
    # weights are (out_features, in_features), scales are (in_features)
    scaled_weight = weight * scales.unsqueeze(0)
    
    # 2. Block-wise quantization (RTN)
    in_features = weight.shape[1]
    dequantized_weight = torch.zeros_like(weight)
    
    for i in range(0, in_features, q_group_size):
        end = min(i + q_group_size, in_features)
        block = scaled_weight[:, i:end]
        
        # Per-output-channel scale for this block
        block_max = block.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        block_scale = block_max / qmax
        
        # Quantize and dequantize
        q_block = torch.round(block / block_scale).clamp(-qmax, qmax)
        dq_block = q_block * block_scale
        
        dequantized_weight[:, i:end] = dq_block
        
    # 3. Un-scale the weights back to their original magnitude scale
    final_weight = dequantized_weight / scales.unsqueeze(0)
    
    return final_weight

@torch.no_grad()
def apply_awq_to_model(model, n_bits: int, calibration_dataloader, skip_layers=['unembed', 'embed'], q_group_size=128):
    """
    Applies simulated AWQ to the HookedTransformer.
    """
    print(f"Applying Simulated AWQ ({n_bits}-bit)...")
    
    # 1. Profile activations to get scales
    act_scales = {}
    
    def get_act_hook(name):
        def hook(tensor, hook=None):
            # tensor is (batch, seq, in_features)
            # we want max magnitude per in_feature
            if name not in act_scales:
                act_scales[name] = tensor.abs().amax(dim=(0, 1))
            else:
                act_scales[name] = torch.max(act_scales[name], tensor.abs().amax(dim=(0, 1)))
        return hook
        
    # Register hooks for inputs to MLPs and Attention
    hooks = []
    for i, block in enumerate(model.blocks):
        hooks.append((f"blocks.{i}.hook_mlp_in", get_act_hook(f"mlp_in_{i}")))
        hooks.append((f"blocks.{i}.hook_resid_pre", get_act_hook(f"attn_in_{i}")))
        
    # Run calibration
    with model.hooks(fwd_hooks=hooks):
        for batch in calibration_dataloader:
            model(batch)
            
    # 2. Apply AWQ
    modified_layers = 0
    for name, param in model.named_parameters():
        if any(skip in name for skip in skip_layers):
            continue
            
        if 'W_in' in name or 'W_out' in name or 'W_q' in name or 'W_k' in name or 'W_v' in name or 'W_o' in name:
            # Figure out which act_scale to use
            block_idx = int(name.split('.')[1])
            if 'mlp' in name:
                scales = act_scales.get(f"mlp_in_{block_idx}")
            else:
                scales = act_scales.get(f"attn_in_{block_idx}")
                
            if scales is None:
                # Fallback to RTN if no scales
                scales = torch.ones(param.shape[-1], device=param.device)
            else:
                # Resize scales if it doesn't match in_features (e.g. W_out takes mlp hidden size)
                if scales.shape[0] != param.shape[-1]:
                     scales = torch.ones(param.shape[-1], device=param.device)
                
            # Treat param as a linear weight
            param.data = simulate_awq_linear(
                nn.Linear(param.shape[-1], param.shape[0], bias=False), 
                n_bits, 
                scales, 
                q_group_size
            )
            modified_layers += 1
            
    print(f"Simulated AWQ applied to {modified_layers} tensors.")
