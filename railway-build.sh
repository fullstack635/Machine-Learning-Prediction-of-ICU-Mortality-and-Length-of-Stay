#!/bin/bash

# Railway build script to handle dependencies properly
set -e

echo "🚀 Starting Railway build process..."

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install wheel and setuptools first
echo "🔧 Installing build tools..."
pip install --no-cache-dir wheel setuptools

# Install requirements with binary-only packages
echo "📚 Installing dependencies..."
pip install --no-cache-dir --only-binary=all -r requirements.txt

echo "✅ Build completed successfully!"
