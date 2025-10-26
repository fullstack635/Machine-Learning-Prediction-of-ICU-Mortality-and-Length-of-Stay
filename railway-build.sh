#!/bin/bash

# Railway build script to handle dependencies properly
set -e

echo "🚀 Starting Railway build process..."

# Check Python version
echo "🐍 Python version:"
python --version

# Ensure Git LFS models are materialized (only if files are LFS pointers)
echo "🔎 Checking for Git LFS pointers in model files..."

is_lfs_pointer() {
  # Return 0 if file looks like a Git LFS pointer
  # Matches either the spec header or the word git-lfs in first 200 bytes
  head -c 200 "$1" 2>/dev/null | grep -q "git-lfs\|^version https://git-lfs.github.com/spec" || return 1
}

materialize_if_lfs() {
  local file="$1"
  if [ -f "$file" ] && is_lfs_pointer "$file"; then
    echo "📦 Detected LFS pointer: $file — installing git-lfs and fetching binary..."
    # Install git-lfs in the build image
    apt-get update -y >/dev/null && apt-get install -y git-lfs >/dev/null
    git lfs install >/dev/null
    # Fetch only required file to keep build fast
    git lfs fetch --include="$file" --exclude="" >/dev/null || true
    git lfs checkout "$file" || true
    if is_lfs_pointer "$file"; then
      echo "❌ Still an LFS pointer after checkout: $file"; ls -lh "$file"; exit 1
    else
      echo "✅ Materialized: $file -> $(stat -c%s "$file" 2>/dev/null || wc -c <"$file") bytes"
    fi
  else
    echo "✅ Not an LFS pointer: $file"
  fi
}

# Run materialization for models we care about
materialize_if_lfs "model.pkl"
materialize_if_lfs "los_lgbm_pipeline.pkl"

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install wheel and setuptools first
echo "🔧 Installing build tools..."
pip install --no-cache-dir wheel setuptools

# Install requirements with binary-only packages using compatible requirements
echo "📚 Installing dependencies with compatible requirements..."
pip install --no-cache-dir --only-binary=all -r requirements-compatible.txt

echo "✅ Build completed successfully!"
