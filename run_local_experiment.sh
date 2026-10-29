#!/bin/bash
set -e
export PYTHONPATH=.:src/circuit_discovery/ACDC

echo "Starting Pythia-410M ACDC Experiment on Local Machine..."

# 1. Run Baseline (FP16)
echo "Running FP16 Baseline..."
venv/bin/python experiments/run_pipeline.py \
    --model EleutherAI/pythia-410m \
    --bits 16 \
    --output fp16_410m_edges.pkl \
    --examples 10

# 2. Run Quantized (8-bit)
echo "Running 8-bit Quantized..."
venv/bin/python experiments/run_pipeline.py \
    --model EleutherAI/pythia-410m \
    --bits 8 \
    --output quant8_410m_edges.pkl \
    --examples 10

# 3. Run Quantized (4-bit)
echo "Running 4-bit Quantized..."
venv/bin/python experiments/run_pipeline.py \
    --model EleutherAI/pythia-410m \
    --bits 4 \
    --output quant4_410m_edges.pkl \
    --examples 10

# 4. Cross-Application Evaluation & Metrics
echo "Evaluating 8-bit Cross-Application..."
venv/bin/python experiments/run_cross_application.py \
    --model EleutherAI/pythia-410m \
    --bits 8 \
    --fp16-edges results/fp16_410m_edges.pkl \
    --quant-edges results/quant8_410m_edges.pkl \
    --examples 10 \
    --json-out results/metrics_410m_sweep.json

echo "Evaluating 4-bit Cross-Application..."
venv/bin/python experiments/run_cross_application.py \
    --model EleutherAI/pythia-410m \
    --bits 4 \
    --fp16-edges results/fp16_410m_edges.pkl \
    --quant-edges results/quant4_410m_edges.pkl \
    --examples 10 \
    --json-out results/metrics_410m_sweep.json

# 5. Plotting Trends
echo "Plotting Trends..."
venv/bin/python experiments/plot_bitwidth_trends.py \
    --json-in results/metrics_410m_sweep.json \
    --output-prefix results/pythia410m_trends

# 6. Visualize 8-bit Overlap Graph
echo "Visualizing 8-bit vs FP16 Network..."
venv/bin/python experiments/visualize_circuits.py \
    --fp16-edges results/fp16_410m_edges.pkl \
    --quant-edges results/quant8_410m_edges.pkl \
    --output results/pythia410m_circuit_overlap_8bit.png

echo "Experiment Complete! Check the 'results/' folder for the metrics and PNGs."
