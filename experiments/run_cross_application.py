import os
import pickle
import torch
import gc
import argparse

from src.tasks.ioi import get_pythia_model, get_ioi_data
from src.quantization.rtn import apply_rtn_to_model
from src.metrics.overlap import compute_jaccard_similarity, compute_precision_recall
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.TLACDCCorrespondence import TLACDCCorrespondence

def load_edges(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_corr_from_edges(exp: TLACDCExperiment, edges: set) -> TLACDCCorrespondence:
    # First, make a copy of the correspondence to avoid mutating the original
    new_corr = TLACDCCorrespondence.setup_from_model(exp.model)
    
    # Iterate through all possible edges and set .present based on the loaded edges
    for t, e in new_corr.all_edges().items():
        if t in edges:
            e.present = True
        else:
            e.present = False
            
    return new_corr

def run_cross_application(model_name, n_bits, fp16_edges_path, quant_edges_path, task="ioi", num_examples=20, device="cpu"):
    fp16_edges = load_edges(fp16_edges_path)
    quant_edges = load_edges(quant_edges_path)
    
    print(f"Loaded {len(fp16_edges)} FP16 edges and {len(quant_edges)} Quantized edges.")
    
    # Compare overlap
    jaccard = compute_jaccard_similarity(fp16_edges, quant_edges)
    precision, recall = compute_precision_recall(fp16_edges, quant_edges)
    
    print("-" * 50)
    print("Circuit Overlap Metrics:")
    print(f"Jaccard Similarity: {jaccard:.4f}")
    print(f"Precision (Quantized edges in FP16): {precision:.4f}")
    print(f"Recall (FP16 edges in Quantized): {recall:.4f}")
    print("-" * 50)
    
    # Load FP16 Model
    print(f"Loading FP16 model {model_name}...")
    fp16_model = get_pythia_model(model_name=model_name, device=device)
    fp16_things = get_ioi_data(fp16_model, num_examples=num_examples, device=device, metric_name="kl_div")
    
    fp16_exp = TLACDCExperiment(
        model=fp16_model,
        threshold=0.0,
        using_wandb=False,
        zero_ablation=True,
        ds=fp16_things.validation_data,
        ref_ds=fp16_things.validation_patch_data,
        metric=fp16_things.validation_metric,
        verbose=False
    )
    
    # Load Quantized Model
    print(f"Loading Quantized model ({n_bits}-bit)...")
    quant_model = get_pythia_model(model_name=model_name, device=device)
    apply_rtn_to_model(quant_model, n_bits, skip_layers=['unembed', 'embed'])
    quant_things = get_ioi_data(quant_model, num_examples=num_examples, device=device, metric_name="kl_div")
    
    quant_exp = TLACDCExperiment(
        model=quant_model,
        threshold=0.0,
        using_wandb=False,
        zero_ablation=True,
        ds=quant_things.validation_data,
        ref_ds=quant_things.validation_patch_data,
        metric=quant_things.validation_metric,
        verbose=False
    )
    
    print("Evaluating cross-applications...")
    
    # 1. Base Metrics
    base_corr_fp16 = build_corr_from_edges(fp16_exp, set(fp16_exp.corr.all_edges().keys()))
    base_fp16_perf = fp16_exp.call_metric_with_corr(base_corr_fp16, fp16_things.validation_metric, fp16_things.validation_data)
    print(f"FP16 Model | Full Circuit Metric: {base_fp16_perf:.4f}")
    
    # 2. fp16 Circuit on FP16 Model
    fp16_corr = build_corr_from_edges(fp16_exp, fp16_edges)
    fp16_on_fp16_perf = fp16_exp.call_metric_with_corr(fp16_corr, fp16_things.validation_metric, fp16_things.validation_data)
    print(f"FP16 Model | FP16 Circuit Metric: {fp16_on_fp16_perf:.4f}")
    
    # 3. Quant Circuit on FP16 Model (Cross)
    quant_corr_on_fp16 = build_corr_from_edges(fp16_exp, quant_edges)
    quant_on_fp16_perf = fp16_exp.call_metric_with_corr(quant_corr_on_fp16, fp16_things.validation_metric, fp16_things.validation_data)
    print(f"FP16 Model | Quant Circuit Metric: {quant_on_fp16_perf:.4f}")
    
    print("-" * 50)
    
    base_corr_quant = build_corr_from_edges(quant_exp, set(quant_exp.corr.all_edges().keys()))
    base_quant_perf = quant_exp.call_metric_with_corr(base_corr_quant, quant_things.validation_metric, quant_things.validation_data)
    print(f"Quant Model | Full Circuit Metric: {base_quant_perf:.4f}")
    
    # 4. Quant Circuit on Quant Model
    quant_corr = build_corr_from_edges(quant_exp, quant_edges)
    quant_on_quant_perf = quant_exp.call_metric_with_corr(quant_corr, quant_things.validation_metric, quant_things.validation_data)
    print(f"Quant Model | Quant Circuit Metric: {quant_on_quant_perf:.4f}")
    
    # 5. fp16 Circuit on Quant Model (Cross)
    fp16_corr_on_quant = build_corr_from_edges(quant_exp, fp16_edges)
    fp16_on_quant_perf = quant_exp.call_metric_with_corr(fp16_corr_on_quant, quant_things.validation_metric, quant_things.validation_data)
    print(f"Quant Model | FP16 Circuit Metric: {fp16_on_quant_perf:.4f}")
    
    print("-" * 50)
    
    return {
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "fp16_on_fp16": fp16_on_fp16_perf,
        "quant_on_fp16": quant_on_fp16_perf,
        "quant_on_quant": quant_on_quant_perf,
        "fp16_on_quant": fp16_on_quant_perf,
    }

if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-14m")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--fp16-edges", type=str, required=True, help="Path to fp16 edge pickle")
    parser.add_argument("--quant-edges", type=str, required=True, help="Path to quant edge pickle")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--json-out", type=str, help="Path to save metrics as JSON")
    args = parser.parse_args()
    
    results = run_cross_application(
        model_name=args.model,
        n_bits=args.bits,
        fp16_edges_path=args.fp16_edges,
        quant_edges_path=args.quant_edges,
        num_examples=args.examples
    )
    
    if args.json_out:
        # Convert float32 tensors to python floats if needed
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                clean_results[k] = v.item()
            else:
                clean_results[k] = v
                
        # Append to existing array or create new
        if os.path.exists(args.json_out):
            with open(args.json_out, "r") as f:
                data = json.load(f)
        else:
            data = {}
            
        if args.model not in data:
            data[args.model] = {}
        
        data[args.model][str(args.bits)] = clean_results
        
        with open(args.json_out, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Appended results to {args.json_out}")
