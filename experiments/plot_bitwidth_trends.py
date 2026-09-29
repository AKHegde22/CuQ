import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_trends(json_path, output_prefix):
    with open(json_path, "r") as f:
        data = json.load(f)
        
    for model_name, bit_results in data.items():
        bits = []
        jaccard = []
        precision = []
        recall = []
        
        fp16_perf = []
        quant_on_quant_perf = []
        quant_on_fp16_perf = []
        fp16_on_quant_perf = []
        
        # Sort bits descending (e.g. 16, 8, 6, 4, 3)
        sorted_bits = sorted([int(b) for b in bit_results.keys()], reverse=True)
        
        for b in sorted_bits:
            res = bit_results[str(b)]
            bits.append(b)
            jaccard.append(res['jaccard'])
            precision.append(res['precision'])
            recall.append(res['recall'])
            
            fp16_perf.append(res['fp16_on_fp16'])
            quant_on_quant_perf.append(res['quant_on_quant'])
            quant_on_fp16_perf.append(res['quant_on_fp16'])
            fp16_on_quant_perf.append(res['fp16_on_quant'])
            
        # Plot 1: Overlap Metrics
        plt.figure(figsize=(10, 6))
        plt.plot(bits, jaccard, marker='o', linewidth=2, label="Jaccard Similarity")
        plt.plot(bits, precision, marker='s', linewidth=2, linestyle='--', label="Precision")
        plt.plot(bits, recall, marker='^', linewidth=2, linestyle='--', label="Recall")
        
        plt.gca().invert_xaxis() # Show bits descending (16 -> 8 -> 4...)
        plt.xlabel("Bit-width (Quantization Level)", fontsize=14)
        plt.ylabel("Overlap Score (0 to 1)", fontsize=14)
        plt.title(f"Circuit Overlap vs Bit-width ({model_name})", fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)
        
        out1 = f"{output_prefix}_overlap.png"
        plt.savefig(out1, bbox_inches='tight', dpi=300)
        print(f"Saved overlap plot to {out1}")
        plt.close()
        
        # Plot 2: Faithfulness Performance
        plt.figure(figsize=(10, 6))
        plt.plot(bits, fp16_perf, marker='o', linewidth=2, color='black', label="FP16 Circuit on FP16 Model (Baseline)")
        plt.plot(bits, quant_on_quant_perf, marker='s', linewidth=2, label="Quantized Circuit on Quantized Model")
        plt.plot(bits, quant_on_fp16_perf, marker='^', linewidth=2, linestyle='--', color='red', label="Quantized Circuit on FP16 Model (Ablated)")
        plt.plot(bits, fp16_on_quant_perf, marker='v', linewidth=2, linestyle='-.', color='blue', label="FP16 Circuit on Quantized Model (Faithfulness)")
        
        plt.gca().invert_xaxis()
        plt.xlabel("Bit-width (Quantization Level)", fontsize=14)
        plt.ylabel("KL Divergence (Lower is better)", fontsize=14)
        plt.title(f"Cross-Application Performance ({model_name})", fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)
        
        out2 = f"{output_prefix}_faithfulness.png"
        plt.savefig(out2, bbox_inches='tight', dpi=300)
        print(f"Saved faithfulness plot to {out2}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-in", type=str, required=True, help="Input JSON file from run_cross_application.py")
    parser.add_argument("--output-prefix", type=str, default="results/trends", help="Prefix for output plot PNGs")
    args = parser.parse_args()
    
    plot_trends(args.json_in, args.output_prefix)
