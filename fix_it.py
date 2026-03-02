from transformers import AutoModelForCausalLM, AutoProcessor
import torch

model_id = "microsoft/Florence-2-base"
print("Connecting to Hugging Face...")

# This script only does one thing: Download the files to your Mac
try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    print("✅ SUCCESS! The model is now saved on your Mac.")
    print("You can now delete this file and run 'streamlit run app.py'")
except Exception as e:
    print(f"❌ Download failed: {e}")