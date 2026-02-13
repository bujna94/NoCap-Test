#!/bin/bash
# =============================================================
# Vast.ai Setup Script - GPT-2 Training Speedup
# Run this on a fresh vast.ai instance to get everything going
# =============================================================
set -e

echo "============================================"
echo "  GPT-2 Training Speedup - Vast.ai Setup"
echo "============================================"

# 1. Clone the repo
REPO_URL="git@github.com:bujna94/NoCap-Test.git"
WORK_DIR="/workspace/NoCap-Test"

if [ -d "$WORK_DIR" ]; then
    echo "[setup] Repo already exists, pulling latest..."
    cd "$WORK_DIR"
    git pull
else
    echo "[setup] Cloning repository..."
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

# 2. Install Python dependencies
echo "[setup] Installing Python dependencies..."
pip install --upgrade pip
pip install anthropic numpy requests huggingface_hub
# Only install PyTorch if not already present (vast.ai images usually have it)
python -c "import torch; print(f'PyTorch {torch.__version__} already installed')" 2>/dev/null || \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 3. Download training data
echo "[setup] Downloading training data (this takes ~30-60 minutes)..."
if [ -d "data/fineweb10B" ] && [ "$(ls -1 data/fineweb10B/*.bin 2>/dev/null | wc -l)" -ge 51 ]; then
    echo "[setup] Data already downloaded (51 files found)"
else
    python data/cached_fineweb10B.py
fi

# 4. Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "============================================"
    echo "  WARNING: ANTHROPIC_API_KEY not set!"
    echo "  Run: export ANTHROPIC_API_KEY=sk-ant-..."
    echo "  Then: python orchestrator.py"
    echo "============================================"
    echo ""
fi

# 5. Reset experiments.json for fresh start
echo "[setup] Resetting experiments.json for fresh run..."
cat > experiments.json << 'EJSON'
{
  "target_val_loss": 3.3821,
  "baseline_time_seconds": null,
  "experiments": []
}
EJSON

# 6. Verify GPU
echo "[setup] GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No GPU detected!"

# 7. Start dashboard + orchestrator
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To start the autonomous system:"
echo "  cd $WORK_DIR"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo "  nohup python dashboard/serve.py &"
echo "  python orchestrator.py"
echo ""
echo "Dashboard will be at: http://<your-vastai-ip>:8080/dashboard/index.html"
echo ""

# Optional: auto-start if API key is set
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "[setup] API key found, starting automatically..."
    # Start dashboard in background
    nohup python dashboard/serve.py > /dev/null 2>&1 &
    echo "[setup] Dashboard started on port 8080"
    # Start orchestrator (foreground so you can see output)
    python orchestrator.py
fi
