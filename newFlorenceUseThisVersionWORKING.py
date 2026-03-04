import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Setup
device = "mps"
model_id = "microsoft/Florence-2-base"

# 2. Load
print(f"🚀 Loading model onto {device}...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

# 3. Path to your image
image_path = "/Users/lorenmannoia/Desktop/test2ImageFlorence.png" 

if os.path.exists(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # Task: Object Detection
    prompt = "<OD>"
    
    print(f"🧠 Analyzing image...")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

    print(f"\n🔍 Results for {os.path.basename(image_path)}:")
    print(parsed_answer)

    # --- 4. VISUALIZATION SECTION ---
    print(f"🎨 Drawing bounding boxes...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    
    # Extract bounding boxes and labels from the results
    # Florence-2 OD output format: {'<OD>': {'bboxes': [[y1, x1, y2, x2], ...], 'labels': ['label', ...]}}
    data = parsed_answer['<OD>']
    
    for bbox, label in zip(data['bboxes'], data['labels']):
        y1, x1, y2, x2 = bbox
        # Create a Rectangle patch (x, y, width, height)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        # Add the label text
        plt.text(x1, y1, label, color='white', weight='bold', verticalalignment='bottom', 
                 bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})
    
    plt.axis('off') # Hide the axis numbers
    
    # Save the result to your Desktop
    output_path = "/Users/lorenmannoia/Desktop/detected_objects.jpg"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved visual result to: {output_path}")
    
    # Show the window (this will pause the script until you close the window)
    plt.show()

else:
    print(f"❌ File not found at: {image_path}")
