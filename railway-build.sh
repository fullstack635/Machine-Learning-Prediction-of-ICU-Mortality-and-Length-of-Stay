#!/bin/bash

# Railway build script to handle dependencies properly
set -e

echo "🚀 Starting Railway build process..."

# Check Python version
echo "🐍 Python version:"
python --version

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install wheel and setuptools first
echo "🔧 Installing build tools..."
pip install --no-cache-dir wheel setuptools

# Install requirements with binary-only packages using stable requirements
echo "📚 Installing dependencies with stable requirements..."
pip install --no-cache-dir --only-binary=all -r requirements-stable.txt

echo "✅ Build completed successfully!"
