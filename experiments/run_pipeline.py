import torch
import argparse
import pickle
import os
import gc

from src.tasks.ioi import get_pythia_model, get_ioi_data
from src.tasks.greater_than import get_greaterthan_data
from src.quantization.rtn import apply_rtn_to_model
from src.quantization.awq import apply_awq_to_model
from src.quantization.gptq import apply_gptq_to_model
from src.metrics.overlap import compute_jaccard_similarity
from acdc.TLACDCExperiment import TLACDCExperiment
from transformer_lens.HookedTransformer import HookedTransformer

def run_acdc_experiment(model_name, n_bits, quantizer="rtn", task="ioi", num_examples=20, threshold=0.01, device="cpu"):
    print(f"Loading model {model_name} on {device}...")
    tl_model = get_pythia_model(model_name=model_name, device=device)
    
    print(f"Loading {task} data...")
    if task == "ioi":
        things = get_ioi_data(tl_model, num_examples=num_examples, device=device, metric_name="kl_div")
    elif task == "greater_than":
        things = get_greaterthan_data(tl_model, num_examples=num_examples, device=device, metric_name="prob_diff")
    else:
        raise NotImplementedError(f"Task {task} not implemented")
        
    if n_bits < 16:
        print(f"Applying {n_bits}-bit {quantizer.upper()} quantization...")
        if quantizer == "rtn":
            apply_rtn_to_model(tl_model, n_bits, skip_layers=['unembed', 'embed'])
        elif quantizer == "awq":
            calib_data = [things.validation_data]
            apply_awq_to_model(tl_model, n_bits, calib_data, skip_layers=['unembed', 'embed'])
        elif quantizer == "gptq":
            calib_data = [things.validation_data]
            apply_gptq_to_model(tl_model, n_bits, calib_data, skip_layers=['unembed', 'embed'])
        else:
            raise ValueError(f"Unknown quantizer: {quantizer}")
        
    tl_model.reset_hooks()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Initializing ACDC experiment...")
    exp = TLACDCExperiment(
        model=tl_model,
        threshold=threshold,
        using_wandb=False,
        zero_ablation=True,
        ds=things.validation_data,
        ref_ds=things.validation_patch_data,
        metric=things.validation_metric,
        verbose=True,
        indices_mode="normal",
        names_mode="normal",
        corrupted_cache_cpu=True,
        hook_verbose=False,
        online_cache_cpu=True,
        add_sender_hooks=True,
        use_pos_embed=False,
        add_receiver_hooks=False,
        remove_redundant=False,
        show_full_index=False,
    )

    print("Running ACDC loop...")
    for i in range(1000): # max steps
        exp.step(testing=False)
        if i % 10 == 0:
            print(f"Step {i}, Edges remaining: {exp.count_no_edges()}")
        if exp.current_node is None:
            break
            
    print(f"Finished. Edges remaining: {exp.count_no_edges()}")
    
    # Extract the discovered edges
    edges_present = set()
    for t, e in exp.corr.all_edges().items():
        if e.present:
            edges_present.add(t)
            
    return edges_present

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-14m") # use 14m for fast local testing
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--task", type=str, default="ioi", choices=["ioi", "greater_than"])
    parser.add_argument("--quantizer", type=str, default="rtn", choices=["rtn", "awq", "gptq"])
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="results.pkl")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    edges = run_acdc_experiment(
        model_name=args.model,
        n_bits=args.bits,
        quantizer=args.quantizer,
        task=args.task,
        num_examples=args.examples,
        threshold=args.threshold,
        device=device
    )
    
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", args.output)
    with open(out_path, "wb") as f:
        pickle.dump(edges, f)
        
    print(f"Saved {len(edges)} edges to {out_path}")
