import torch
from src.quantization.rtn import quantize_rtn_symmetric, apply_rtn_to_model

def test_quantize_rtn_symmetric_8bit():
    weight = torch.randn(10, 20)
    q_weight = quantize_rtn_symmetric(weight, 8)
    
    assert q_weight.shape == weight.shape
    assert q_weight.dtype == weight.dtype
    
    # Error should be relatively small for 8-bit
    mse = torch.nn.functional.mse_loss(weight, q_weight)
    assert mse < 0.05

def test_quantize_rtn_symmetric_4bit():
    weight = torch.randn(10, 20)
    q_weight = quantize_rtn_symmetric(weight, 4)
    
    assert q_weight.shape == weight.shape
    assert q_weight.dtype == weight.dtype
    
    # Error will be larger for 4-bit, but bounded
    mse = torch.nn.functional.mse_loss(weight, q_weight)
    assert mse < 0.5  # Typical MSE for 4-bit normal distribution

def test_apply_rtn_to_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, 10)
    )
    
    original_weight = model[0].weight.clone()
    apply_rtn_to_model(model, 4)
    
    new_weight = model[0].weight
    
    # The weight should have changed (due to quantization error)
    assert not torch.allclose(original_weight, new_weight)
    
    # It should still be the same shape and type
    assert new_weight.shape == original_weight.shape
    assert new_weight.dtype == original_weight.dtype
