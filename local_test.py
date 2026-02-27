# OKAY SO BASICALLY THIS FILE IS TO RUN THE TEST IMAGE HERE IT BASICALLY LOOKS AT THE IMAGE IN UR FOLDER AND GIVES THE FILE TO THE MODEL

# okay i know the next few lines look scary like its gonna download a bug onto ur computer but i promise it wont bc it just installs the stuff uneed to run it
import torch
from PIL to import Image
from transformers import AutoProcessor, AutoModelForCasualLM

# --- CONFIGURATION ---
device = "mps" if torch.backends.mps.is_available() else "cpu" # checks if ure using a mac silicon model which is mps and defaults to cpu if not
model_id = "microsoft/Florence-2-base" # downloads florence-2
image_path = "my_test_image.jpg"  # IMPORTANT GUYS MAKE SURE U ADD AN IMAGE TO UR FOLDER W ALL THIS CODE IN IT AND NAME IT "my_test_image.jpg" AND PUT IT IN THOSE QUOTES

print(f"🚀 Loading model onto {device}...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval() # "trust_remote_code basically tells the computer that we're not downloading a virus and .eval() tells florence to "evaluate" the image
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def ask_florence(prompt, img): # THIS IS THE CODE THAT RUNS FLORENCE
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(device) # converts prompt and and image into mathematical models that florence can understand
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3 # basically tells it to explore three different word paths at once and choose the best one
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task=prompt, image_size=(img.width, img.height))
    return result[prompt]

# --- THE TEST ---
try: # OKAT THIS IS THE MEAT OF THE CODE THIS GIVES THE OUTPUT
    image = Image.open(image_path).convert("RGB")
    
    print("\n🔍 Task 1: Basic Caption")
    print(ask_florence("<CAPTION>", image)) # <CAPTION> is a prebuilt prompt that tells florence to give a caption

    print("\n🔍 Task 2: Detailed Caption")
    print(ask_florence("<DETAILED_CAPTION>", image))

    print("\n🔍 Task 3: Finding Objects (OCR)")
    print(ask_florence("<OCR>", image))

    print("\n🔍 Task 4: Object Detection")
    od_results = ask_florence("<OD>", image)
    print(od_results)

    # --- VISUALIZING THE BOXES ---
    # This part draws the boxes directly onto your image so you can see them!
    draw = ImageDraw.Draw(image)
    for bbox, label in zip(od_results['bboxes'], od_results['labels']):
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1]), label, fill="red")
    
    image.show() # This opens the image with red boxes around objects

except FileNotFoundError:
    print(f"❌ Error: Could not find the file '{image_path}'. Make sure it's in the same folder as this script!")
