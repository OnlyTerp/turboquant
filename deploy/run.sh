#!/bin/bash
# TerpBot Pro + TurboQuant — Quick deploy on 5090
# Run this on the 5090 PC

set -e

echo "=== TerpBot Pro + TurboQuant Deploy ==="
echo ""

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi not found"

# Set HF token (needed for model download)
if [ -z "$HF_TOKEN" ]; then
    echo "Set your HuggingFace token:"
    echo "  export HF_TOKEN=hf_your_token_here"
    echo ""
    echo "Or create a .env file with:"
    echo "  HF_TOKEN=hf_your_token_here"
    echo ""
fi

# Build and run
echo "Building Docker image..."
docker compose build

echo ""
echo "Starting TerpBot Pro with TurboQuant..."
docker compose up -d

echo ""
echo "Waiting for model to load (this takes 1-2 min on first run)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "=== READY! ==="
        echo ""
        echo "Test it:"
        echo '  curl http://localhost:8000/v1/chat/completions \'
        echo '    -H "Content-Type: application/json" \'
        echo '    -d '"'"'{"model":"terpbot-pro","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'"'"''
        echo ""
        echo "Logs: docker compose logs -f"
        exit 0
    fi
    sleep 2
    printf "."
done

echo ""
echo "Still loading... check: docker compose logs -f"
