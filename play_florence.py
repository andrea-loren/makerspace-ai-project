import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForCausalLM

# 1. Setup - Use Mac's GPU (MPS) if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "microsoft/Florence-2-base"

print(f"⌛ Loading Florence-2 onto {device}... (First time takes a minute)")

# 2. Load the model (We bypass the 'flash_attn' error common on Macs here)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 3. Grab a test image from the web
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 4. Ask the model to describe the image
prompt = "<DETAILED_CAPTION>"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

print("🧠 Thinking...")
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3
)

# 5. Show the result
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

print("\n✨ FLORENCE SAYS:")
print(parsed_answer[prompt])
