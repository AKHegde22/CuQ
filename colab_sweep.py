import os
import subprocess
from google.colab import drive

# 1. Mount Google Drive
print("Mounting Google Drive to /content/drive ...")
drive.mount('/content/drive')

# 2. Setup Save Directory
# All results will be saved securely to your Google Drive. 
# Even if Colab disconnects, whatever finished before the disconnect is safe.
SAVE_DIR = "/content/drive/MyDrive/CuQ_Results"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Results will be saved to: {SAVE_DIR}")

# 3. Setup Python Path
os.environ["PYTHONPATH"] = ".:src/circuit_discovery/ACDC"

# 4. Define the Sweep Matrix
MODELS = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b"
]

TASKS = ["ioi", "greater_than"]
QUANTIZERS = ["rtn", "awq", "gptq"]
BIT_WIDTHS = [8, 6, 4, 3]

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"WARNING: Command failed with return code {result.returncode}")

# 5. Execute the Sweep
for model in MODELS:
    m_name = model.split("/")[-1]
    
    for task in TASKS:
        # Step A: Find the Baseline FP16 Circuit
        fp16_out = f"{SAVE_DIR}/{m_name}_fp16_{task}.pkl"
        if not os.path.exists(fp16_out):
            print(f"\n[{m_name} | {task}] Finding FP16 Baseline Circuit...")
            run_cmd(f"python experiments/run_pipeline.py --model {model} --bits 16 --task {task} --output {fp16_out}")
            
        # Step B: Run Quantization Sweeps
        for quant in QUANTIZERS:
            for bits in BIT_WIDTHS:
                quant_out = f"{SAVE_DIR}/{m_name}_{quant}{bits}_{task}.pkl"
                metrics_out = f"{SAVE_DIR}/{m_name}_{quant}{bits}_{task}_metrics.json"
                
                if os.path.exists(metrics_out):
                    print(f"Skipping {quant_out}, metrics already exist.")
                    continue
                    
                print(f"\n[{m_name} | {task} | {quant.upper()} {bits}-bit] Running...")
                
                # Find Quantized Circuit
                if not os.path.exists(quant_out):
                    run_cmd(f"python experiments/run_pipeline.py --model {model} --bits {bits} --quantizer {quant} --task {task} --output {quant_out}")
                
                # Cross-Application Evaluation
                if os.path.exists(fp16_out) and os.path.exists(quant_out):
                    run_cmd(f"python experiments/run_cross_application.py --model {model} --bits {bits} --quantizer {quant} --task {task} --fp16-edges {fp16_out} --quant-edges {quant_out} --json-out {metrics_out}")

print("\n🎉 Full Sweep Completed Successfully! All results are saved in Google Drive.")
