import streamlit as st
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

# Initialize models
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Image generation pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)

    # Segmentation model
    segmenter = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-small",
        torch_dtype=torch.float32
    ).to(device)
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")

    return pipe, segmenter, image_processor, device

# Streamlit UI
st.title("Satellite Image Processor")

# Navigation
app_mode = st.sidebar.radio(
    "Choose Functionality",
    ["Generate Image", "Segment Image"]
)

# Load models
pipe, segmenter, image_processor, device = load_models()

if app_mode == "Generate Image":
    st.header("Generate Satellite Image")
    prompt = st.text_input("Enter prompt:", "A satellite view of a coastal city")
    if st.button("Generate"):
        with st.spinner("Generating image..."):
            generator = torch.Generator(device=device).manual_seed(42)
            image = pipe(
                prompt=prompt,
                num_inference_steps=30,
                generator=generator,
                guidance_scale=7.5
            ).images[0]
            st.image(image, caption="Generated Satellite Image", use_column_width=True)

elif app_mode == "Segment Image":
    st.header("Land Cover Segmentation")
    uploaded_file = st.file_uploader("Upload satellite image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        if st.button("Perform Segmentation"):
            with st.spinner("Analyzing land cover..."):
                inputs = image_processor(image.convert("RGB"), return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = segmenter(**inputs)
                seg_map = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
                color_palette = np.array([
                    [0, 0, 0],        # Background
                    [34, 139, 34],    # Vegetation
                    [255, 165, 0],    # Urban
                    [0, 0, 255],      # Water
                    [255, 255, 0]     # Barren Land
                ], dtype=np.uint8)
                colored = color_palette[seg_map]
                st.image(colored, caption="Segmentation Map", use_column_width=True)
                st.markdown("**Legend:**")
                st.markdown("- Black: Background\n- Green: Vegetation\n- Orange: Urban\n- Blue: Water\n- Yellow: Barren Land")
