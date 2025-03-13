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

# Load models with caching for performance
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load image generation pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype,
        safety_checker=None
    )
    
    # Device-specific optimizations
    if device == "cuda":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to(device)
    else:
        st.warning("Using CPU mode - expect slower performance. Consider GPU acceleration.")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++"
        )
    
    # Load segmentation model
    segmenter = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-tiny",
        torch_dtype=torch_dtype
    ).to(device)
    
    return pipe, segmenter, AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny"), device

# Initialize models
pipe, segmenter, image_processor, device = load_models()

# Streamlit UI Configuration
st.set_page_config(page_title="Satellite Processor", layout="wide")
st.title("🌍 Satellite Image Processing Suite")

# Sidebar with Settings
with st.sidebar:
    st.header("Settings")
    generate_steps = st.slider("Generation Steps", 15, 30, 20, 
                             help="Fewer steps = faster but less detailed")
    seed = st.number_input("Random Seed", value=42,
                         help="Change for different variations")
    st.markdown("---")
    st.caption(f"Running on: {device.upper()}")

# Main Tabs
tab1, tab2 = st.tabs(["Generate", "Analyze"])

# Image Generation Tab
with tab1:
    st.header("Generate Satellite Imagery")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        prompt = st.text_area("Description:", 
                            "High-resolution satellite view of coastal city with modern infrastructure and green parks",
                            height=100)
        
        if st.button("Generate Image", use_container_width=True):
            with st.spinner(f"Generating ({generate_steps} steps)..."):
                try:
                    generator = torch.Generator(device).manual_seed(int(seed))
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=int(generate_steps),
                        guidance_scale=7.5,
                        generator=generator
                    ).images[0]
                    st.session_state.generated_image = image
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
    
    with col2:
        if "generated_image" in st.session_state:
            st.image(st.session_state.generated_image, 
                   caption="Generated Satellite Image",
                   use_column_width=True)
            st.download_button("Download Image", 
                             Image.fromarray(np.array(st.session_state.generated_image)),
                             file_name="generated_satellite.png")

# Analysis Tab
with tab2:
    st.header("Land Cover Analysis")
    uploaded_file = st.file_uploader("Upload satellite image", 
                                   type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Land Cover", type="primary"):
                with st.spinner("Processing..."):
                    # Downsample large images for CPU efficiency
                    if image.size[0] > 1024 or image.size[1] > 1024:
                        image = image.resize((512, 512))
                        st.warning("Large image downsampled to 512px for memory efficiency")
                    
                    inputs = image_processor(image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = segmenter(**inputs)
                    
                    seg_map = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
                    
                    # Create color overlay
                    color_palette = np.array([
                        [0, 0, 0],        # Background
                        [34, 139, 34],    # Vegetation
                        [255, 165, 0],    # Urban
                        [0, 0, 255],      # Water
                        [255, 255, 0]     # Barren Land
                    ], dtype=np.uint8)
                    
                    colored_mask = color_palette[seg_map.astype(np.uint8)]
                    st.image(colored_mask, 
                           caption="Land Cover Analysis",
                           use_column_width=True,
                           clamp=True)
                    
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

# System Info Footer
st.divider()
st.caption(f"System: {device.upper()} | Torch: {torch.__version__} | Streamlit: {st.__version__}")
