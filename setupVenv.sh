#!/bin/bash

echo "🗑️  Cleaning up old environment..."
rm -rf .venv
rm -rf ~/.cache/huggingface/modules/transformers_modules/microsoft

echo "🏗️  Creating fresh Virtual Environment with Python 3.12..."
# We use python3.12 specifically to avoid the 3.14 issues
python3.12 -m venv .venv

echo "🔌 Activating environment..."
source .venv/bin/activate

echo "⬆️  Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing AI Core (Torch & Transformers)..."
pip install torch torchvision torchaudio "transformers<4.46.0" accelerate

echo "🧠 Installing Florence-2 Dependencies..."
pip install einops timm protobuf sentencepiece

echo "🖼️  Installing Utility & Visualization Tools..."
pip install requests Pillow matplotlib

echo "✅ Setup Complete!"
echo "To start working, run: source .venv/bin/activate"
