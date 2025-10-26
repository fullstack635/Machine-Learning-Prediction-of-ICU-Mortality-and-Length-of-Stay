#!/usr/bin/env python3
"""
Test script to verify model files exist and can be loaded
"""
import os
import joblib
import sys

def test_model_file(model_path, model_name):
    """Test if a model file exists and can be loaded"""
    print(f"\n🔍 Testing {model_name}...")
    print(f"Looking for: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"✅ {model_name} loaded successfully")
            print(f"Model type: {type(model)}")
            
            if hasattr(model, 'feature_names_in_'):
                print(f"Features: {len(model.feature_names_in_)}")
            if hasattr(model, 'n_features_in_'):
                print(f"Expected features: {model.n_features_in_}")
                
            return True
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return False
    else:
        print(f"❌ {model_name} file not found")
        return False

def main():
    print("🧪 Testing Model Files...")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Test primary model
    primary_model_path = "model.pkl"
    primary_loaded = test_model_file(primary_model_path, "Primary Model")
    
    # Test LOS model
    los_model_path = "los_lgbm_pipeline.pkl"
    los_loaded = test_model_file(los_model_path, "LOS Model")
    
    print(f"\n📊 Summary:")
    print(f"Primary Model: {'✅ Loaded' if primary_loaded else '❌ Failed'}")
    print(f"LOS Model: {'✅ Loaded' if los_loaded else '❌ Failed'}")
    
    if not primary_loaded and not los_loaded:
        print("\n⚠️  No models loaded! Please upload model files to Railway.")
        sys.exit(1)
    elif not primary_loaded:
        print("\n⚠️  Primary model not loaded!")
    elif not los_loaded:
        print("\n⚠️  LOS model not loaded!")
    else:
        print("\n🎉 All models loaded successfully!")

if __name__ == "__main__":
    main()
