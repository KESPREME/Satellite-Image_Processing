import sys
import streamlit as st
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

# Fix for Hugging Face Hub compatibility
sys.modules['huggingface_hub'].cached_download = hf_hub_download

# Memory-efficient model loading
@st.cache_resource(show_spinner=False)
def load_generation_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableDiffusionPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    )
    model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
    model.enable_xformers_memory_efficient_attention()
    model = model.to(device)
    model.enable_model_cpu_offload()
    return model

@st.cache_resource(show_spinner=False)
def load_segmentation_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-tiny",
        torch_dtype=torch.float16
    )
    processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
    return model.to(device), processor

# Streamlit UI
st.set_page_config(page_title="Satellite Processor", layout="wide")
st.title("🌍 Satellite Image Processor")

# Lazy-load models only when needed
if "generator" not in st.session_state:
    with st.spinner("Initializing system (this might take a minute)..."):
        st.session_state.generator = load_generation_model()
        st.session_state.segmenter, st.session_state.processor = load_segmentation_model()

# Main Tabs
tab1, tab2 = st.tabs(["Generate", "Analyze"])

with tab1:
    st.header("Generate Satellite Imagery")
    prompt = st.text_area("Description:", "High-resolution satellite view of coastal city", height=100)
    steps = st.slider("Generation Steps", 15, 30, 20)
    
    if st.button("Generate"):
        with st.spinner(f"Generating ({steps} steps)..."):
            try:
                image = st.session_state.generator(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=7.5
                ).images[0]
                st.image(image, use_column_width=True)
                st.session_state.last_image = image
            except torch.cuda.OutOfMemoryError:
                st.error("Memory limit exceeded! Try fewer steps or smaller resolution")

with tab2:
    st.header("Land Cover Analysis")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
        
        if st.button("Analyze"):
            with st.spinner("Processing..."):
                try:
                    # Reduce memory usage by resizing first
                    processed_image = st.session_state.processor(image.resize((256, 256)), return_tensors="pt").to(st.session_state.segmenter.device)
                    with torch.no_grad():
                        outputs = st.session_state.segmenter(**processed_image)
                    
                    seg_map = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
                    
                    # Efficient color mapping
                    colors = np.array([
                        [0,0,0], [34,139,34], [255,165,0], 
                        [0,0,255], [255,255,0]
                    ], dtype=np.uint8)
                    
                    st.image(colors[seg_map.astype(np.uint8)], 
                           caption="Land Cover Analysis",
                           use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

# System Info
st.caption(f"Running on: {'GPU' if torch.cuda.is_available() else 'CPU'} | Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB used")
