# Add this at the very top
import sys
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    !pip install huggingface_hub==0.15.1 --quiet
    from huggingface_hub import hf_hub_download

# Monkey-patch cached_download
sys.modules['huggingface_hub'].cached_download = hf_hub_download

import streamlit as st
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline, DPMSolverMultistepScheduler
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

# Initialize the Satellite Diffusion System
class SatelliteDiffusionSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_models()

    def _initialize_models(self):
        """Initialize models with stable version configuration"""
        # Base generation model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        # Super-resolution model
        self.upscaler = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # Segmentation model
        self.segmenter = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small",
            torch_dtype=torch.float32
        ).to(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")

    def generate_satellite_image(self, prompt, steps=30):
        """Generate 512x512 satellite image"""
        generator = torch.Generator(device=self.device).manual_seed(42)
        return self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=7.5
        ).images[0]

    def enhance_resolution(self, image):
        """4x super-resolution enhancement"""
        lr_image = image.resize((256, 256))
        return self.upscaler(
            prompt="high-resolution satellite imagery",
            image=lr_image,
            num_inference_steps=25,
            guidance_scale=7.5
        ).images[0]

    def semantic_segmentation(self, image):
        """Land cover segmentation"""
        inputs = self.image_processor(image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.segmenter(**inputs)
            
        seg_map = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
        return self._colorize_segmentation(seg_map)

    def _colorize_segmentation(self, seg_map):
        """Generate color-coded segmentation map"""
        color_palette = np.array([
            [0, 0, 0],        # Background
            [34, 139, 34],    # Vegetation
            [255, 165, 0],    # Urban
            [0, 0, 255],      # Water
            [255, 255, 0]     # Barren Land
        ], dtype=np.uint8)
        return Image.fromarray(color_palette[seg_map])

# Streamlit UI
st.set_page_config(page_title="Satellite Image Processor", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Functionality", 
                           ["Generate Image", "Enhance Resolution", "Semantic Segmentation"])

# Initialize system (cached)
@st.cache_resource
def load_system():
    return SatelliteDiffusionSystem()

system = load_system()

# Main content area
if app_mode == "Generate Image":
    st.header("Generate Satellite Image")
    prompt = st.text_input("Enter prompt:", "A satellite view of a coastal city with modern infrastructure")
    if st.button("Generate"):
        with st.spinner("Generating satellite image..."):
            image = system.generate_satellite_image(prompt)
            st.image(image, caption="Generated Satellite Image", use_column_width=True)

elif app_mode == "Enhance Resolution":
    st.header("Enhance Image Resolution")
    uploaded_file = st.file_uploader("Upload low-res image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        if st.button("Enhance Resolution"):
            with st.spinner("Enhancing image..."):
                enhanced = system.enhance_resolution(image)
                st.image(enhanced, caption="Enhanced Image", use_column_width=True)

elif app_mode == "Semantic Segmentation":
    st.header("Land Cover Segmentation")
    uploaded_file = st.file_uploader("Upload satellite image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        if st.button("Perform Segmentation"):
            with st.spinner("Analyzing land cover..."):
                seg_map = system.semantic_segmentation(image)
                st.image(seg_map, caption="Segmentation Map", use_column_width=True)
                st.markdown("**Legend:**")
                st.markdown("- Black: Background\n- Green: Vegetation\n- Orange: Urban\n- Blue: Water\n- Yellow: Barren Land")
