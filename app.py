import streamlit as st
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import pandas as pd
import torch
import os

# SET PAGE CONFIG
st.set_page_config(page_title="Makerspace Assistant", layout="centered")

# 1. SETUP THE BRAIN (Florence-2)
@st.cache_resource
def load_model():
    model_id = "microsoft/Florence-2-base"
    # Mac Optimization
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Adding a status spinner so you know it's working
    with st.spinner("Loading AI Brain... (This takes a minute on first run)"):
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device

# 2. THE MAP LOOKUP (Using Pandas for better Mac support)
def find_in_inventory(item_name):
    if not os.path.exists('inventory.csv'):
        st.error("Missing inventory.csv file!")
        return None
    
    df = pd.read_csv('inventory.csv')
    # Search for the detected label inside our CSV
    match = df[df['item'].str.contains(item_name.lower(), case=False, na=False)]
    
    if not match.empty:
        return match.iloc[0]['location']
    return None

# START APP
st.title("🛠️ Makerspace Assistant")
st.write("Snap a photo to find out where a tool belongs.")

model, processor, device = load_model()

img_file = st.camera_input("Take a photo of the tool")

if img_file:
    image = Image.open(img_file).convert("RGB")
    
    # 3. ADVANCED PROMPT (The "Admin" way to differentiate scissors)
    # We tell the AI exactly what to look for in a cluttered room
    target_items = "sewing scissors, regular scissors, hammer, pliers, screwdriver, tape measure"
    prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {target_items}"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with st.spinner("Identifying tool..."):
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task="<CAPTION_TO_PHRASE_GROUNDING>", 
        image_size=(image.width, image.height)
    )

    # 4. RESULTS DISPLAY
    # The dictionary key for grounding is different than basic OD
    results = parsed_answer.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
    labels = results.get('labels', [])

    if labels:
        # Get unique labels only
        for label in set(labels):
            location = find_in_inventory(label)
            if location:
                st.balloons()
                st.success(f"✅ Found: **{label.upper()}**")
                st.info(f"📍 Destination: **{location}**")
            else:
                st.warning(f"🔍 I saw '{label}', but I don't have a location for it in the CSV.")
    else:
        st.error("I couldn't identify any known tools. Try holding it closer to the camera!")
