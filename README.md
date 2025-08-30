# Satellite Image Processing Web Application

[![GitHub Workflow Status](https://github.com/KESPREME/Satellite-Image_Processing/actions/workflows/main.yml/badge.svg)](https://github.com/KESPREME/Satellite-Image_Processing/actions)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange.svg)](https://streamlit.io/)

This project provides a user-friendly Streamlit web application for processing satellite images using pre-trained models from the Hugging Face Hub.  Leveraging the power of Hugging Face's model ecosystem, this application allows for efficient and accessible image analysis.

## Features

* **Satellite Image Upload:** Easily upload your satellite images for processing.
* **Model Selection:** Choose from a range of pre-trained Hugging Face models tailored for various satellite image processing tasks (e.g., classification, segmentation, generation).
* **Real-time Processing:**  View the processed images instantly within the Streamlit interface.
* **Hugging Face Model Integration:**  Benefit from the continuously expanding library of state-of-the-art models on Hugging Face.
* **Intuitive User Interface:** A streamlined and user-friendly experience designed for both novice and experienced users.

## Technologies Used

* **Python:** The primary programming language.
* **Streamlit:**  The framework for building the interactive web application.
* **Hugging Face Transformers & Diffusers:** Libraries for accessing and utilizing pre-trained models for image processing.
* **xformers:** Library for optimizing transformer model performance, enabling faster processing of larger images.

## Installation

1. **Clone the repository:**
   bash
   git clone https://github.com/KESPREME/Satellite-Image_Processing.git
   cd Satellite-Image_Processing
   
2. **Create a virtual environment (recommended):**
   bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
3. **Install dependencies:**
   bash
   pip install -r requirements.txt
   

## Usage

1. Run the Streamlit application:
   bash
   streamlit run app.py
   
2.  The application will open in your web browser. Follow the on-screen instructions to upload your satellite image and select a processing model.
3. The processed image will be displayed in real-time.

## Project Structure


Satellite-Image_Processing/
├── app.py             # Main application script
├── requirements.txt   # Project dependencies
└── .github/
    └── workflows/     # GitHub Actions workflow files
└── .devcontainer/     # Development container configuration


## Contributing

Contributions are welcome! Please open an issue to discuss potential features or bug fixes before submitting a pull request.  Ensure your code adheres to the PEP 8 style guide and includes comprehensive tests.

## License

This project is currently unlicensed.  A license will be added to this repository in the future.
