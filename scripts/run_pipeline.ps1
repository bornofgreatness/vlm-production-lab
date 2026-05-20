# Full VLM post-training pipeline (requires NVIDIA GPU + CUDA PyTorch)
# Run from repo root: .\scripts\run_pipeline.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "==> 1. Curate ScienceQA subset"
python -m src.data.curate --max-samples 512 --out-dir data/curated

Write-Host "==> 2. LoRA SFT (Qwen2-VL-2B, 4-bit)"
python -m src.train.sft_lora --config configs/sft_lora.yaml

Write-Host "==> 3. Baseline eval"
python -m src.eval.benchmarks --config configs/eval.yaml --adapter ""

# Copy baseline to results/eval_baseline.json manually or re-run with no adapter

Write-Host "==> 4. Finetuned eval"
python -m src.eval.benchmarks --config configs/eval.yaml

Write-Host "==> 5. Efficiency benchmark"
python -m src.deploy.efficiency_benchmark

Write-Host "Done. Check results/ and outputs/"
