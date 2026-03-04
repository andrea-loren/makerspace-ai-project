import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Setup
device = "mps"
model_id = "microsoft/Florence-2-base"

print(f"🚀 Loading model onto {device}...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

def run_example(task_prompt, image):
    """General function to handle different Florence-2 tasks."""
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return result

def draw_results(image, od_results):
    """Draws the bounding boxes if Object Detection was used."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    data = od_results['<OD>']
    for bbox, label in zip(data['bboxes'], data['labels']):
        y1, x1, y2, x2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', weight='bold', bbox={'facecolor': 'red', 'alpha': 0.5})
    plt.axis('off')
    plt.show()

# --- MAIN EXECUTION ---
image_path = "/Users/lorenmannoia/Desktop/testImageFlorence.jpg" 

if os.path.exists(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    
    # FUNCTION 1: Generate a Detailed Caption
    print("📝 Generating caption...")
    caption_result = run_example("<DETAILED_CAPTION>", raw_image)
    print(f"\nCaption: {caption_result['<DETAILED_CAPTION>']}")

    # FUNCTION 2: Object Detection
    print("\n🎯 Running Object Detection...")
    od_result = run_example("<OD>", raw_image)
    print(f"Detection Results: {od_result}")
    
    # Visualize the detection
    draw_results(raw_image, od_result)

else:
    print(f"❌ File not found at: {image_path}")
