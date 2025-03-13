# Fix for cached_download import error
import sys
from huggingface_hub import hf_hub_download
sys.modules['huggingface_hub'].cached_download = hf_hub_download

# Import remaining libraries
import streamlit as st
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

# Load models with caching for performance
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 for NVIDIA GPUs
    
    # Load optimized image generation pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    
    # Load lightweight segmentation model
    segmenter = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-tiny",
        torch_dtype=torch.float16
    ).to(device)
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
    
    return pipe, segmenter, image_processor, device

# Initialize models
pipe, segmenter, image_processor, device = load_models()

# Streamlit UI Configuration
st.set_page_config(page_title="Satellite Processor", layout="wide")
st.title("🌍 Satellite Image Processing Suite")

# Sidebar Settings
with st.sidebar:
    st.header("Settings")
    generate_steps = st.slider("Generation Steps", 20, 50, 25)
    seed = st.number_input("Random Seed", value=42)
    st.markdown("---")
    st.caption(f"Running on: {device.upper()}")

# Main Tabs
tab1, tab2 = st.tabs(["Generate", "Analyze"])

# Image Generation Tab
with tab1:
    st.header("Generate Satellite Imagery")
    prompt = st.text_area("Description:", 
                          "High-resolution satellite view of a coastal city with modern infrastructure",
                          height=100)
    
    if st.button("Generate Image", type="primary"):
        with st.spinner(f"Generating (est. 15-30s on {device.upper()})..."):
            try:
                generator = torch.Generator(device).manual_seed(int(seed))
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=int(generate_steps),
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]
                st.session_state.generated_image = image
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

# Image Analysis Tab
with tab2:
    st.header("Land Cover Analysis")
    uploaded_file = st.file_uploader("Upload satellite image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Land Cover", type="primary"):
                with st.spinner(f"Processing (est. 5-10s on {device.upper()})..."):
                    inputs = image_processor(image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = segmenter(**inputs)
                    
                    seg_map = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
                    
                    # Color palette for segmentation
                    color_palette = np.array([
                        [0, 0, 0],        # Background
                        [34, 139, 34],    # Vegetation
                        [255, 165, 0],    # Urban
                        [0, 0, 255],      # Water
                        [255, 255, 0]     # Barren Land
                    ], dtype=np.uint8)
                    
                    colored_mask = color_palette[seg_map.astype(np.uint8)]  # Ensure integer indices
                    st.image(colored_mask, caption="Land Cover Analysis", use_column_width=True, clamp=True)
                    
                    # Legend
                    with st.expander("Color Legend"):
                        st.markdown("""
                        - ⬛ Background
                        - 🟩 Vegetation
                        - 🟧 Urban Areas
                        - 🟦 Water Bodies
                        - 🟨 Barren Land
                        """)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Footer
st.divider()
st.caption(f"System: {device.upper()} | Torch: {torch.__version__} | Streamlit: {st.__version__}")
