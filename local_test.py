# OKAY SO BASICALLY THIS FILE IS TO RUN THE TEST IMAGE HERE IT BASICALLY LOOKS AT THE IMAGE IN UR FOLDER AND GIVES THE FILE TO THE MODEL

# --- CONFIGURATION ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "microsoft/Florence-2-base"
image_path = "my_test_image.jpg"  # IMPORTANT GUYS MAKE SURE U ADD AN IMAGE TO UR FOLDER W ALL THIS CODE IN IT AND NAME IT "my_test_image.jpg" AND PUT IT IN THOSE QUOTES

print(f"🚀 Loading model onto {device}...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def ask_florence(prompt, img):
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task=prompt, image_size=(img.width, img.height))
    return result[prompt]

# --- THE TEST ---
try:
    image = Image.open(image_path).convert("RGB")
    
    print("\n🔍 Task 1: Basic Caption")
    print(ask_florence("<CAPTION>", image))

    print("\n🔍 Task 2: Detailed Caption")
    print(ask_florence("<DETAILED_CAPTION>", image))

    print("\n🔍 Task 3: Finding Objects (OCR)")
    print(ask_florence("<OCR>", image))

except FileNotFoundError:
    print(f"❌ Error: Could not find the file '{image_path}'. Make sure it's in the same folder as this script!")
