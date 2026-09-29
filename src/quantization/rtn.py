import torch

def quantize_rtn_symmetric(weight: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Performs simulated per-channel symmetric Round-To-Nearest (RTN) quantization.
    
    Args:
        weight (torch.Tensor): The fp16/fp32 weight tensor to quantize. 
                               Assumes shape (out_features, in_features) for linear layers.
                               The quantization is performed over the last dimension (per-output-channel).
        n_bits (int): The target bit-width (e.g., 8, 6, 4, 3).
        
    Returns:
        torch.Tensor: The dequantized weight tensor in the original dtype, simulating quantization error.
    """
    if n_bits >= 16:
        return weight.clone()
        
    # Calculate qmax (e.g., for 4-bit, qmax = 2^3 - 1 = 7)
    qmax = (1 << (n_bits - 1)) - 1
    
    # Per-channel maximum absolute value
    # We assume the last dimension is the input dimension, and we quantize per output channel.
    # For linear layers (out_dim, in_dim), this computes max over in_dim.
    # If the tensor is 1D (e.g., biases), we just quantize it globally or per-element (max over dim=0).
    if weight.dim() > 1:
        max_val = weight.abs().amax(dim=-1, keepdim=True)
    else:
        max_val = weight.abs().amax(keepdim=True)
        
    # Prevent division by zero
    scale = max_val.clamp(min=1e-8) / qmax
    
    # Quantize
    q_weight = torch.round(weight / scale)
    
    # Clamp to valid range (symmetric: -qmax to qmax)
    q_weight = torch.clamp(q_weight, -qmax, qmax)
    
    # Dequantize (simulated quantization)
    dequantized_weight = q_weight * scale
    
    return dequantized_weight.to(weight.dtype)

def apply_rtn_to_model(model: torch.nn.Module, n_bits: int, skip_layers: list = None):
    """
    Applies simulated RTN quantization to all Linear and Embedding layers in a model in-place.
    
    Args:
        model (torch.nn.Module): The PyTorch model to quantize.
        n_bits (int): Target bit-width.
        skip_layers (list): List of layer names to skip (e.g., ['unembed', 'embed']).
    """
    if skip_layers is None:
        skip_layers = []
        
    for name, module in model.named_modules():
        if any(skip_name in name for skip_name in skip_layers):
            continue
            
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            with torch.no_grad():
                module.weight.copy_(quantize_rtn_symmetric(module.weight, n_bits))
