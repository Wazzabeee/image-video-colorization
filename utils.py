import numpy as np
import requests
import streamlit as st
from PIL import Image

from models.deep_colorization.colorizers import postprocess_tens, preprocess_img, load_img, eccv16, siggraph17


# Define a function that we can use to load lottie files from a link.
@st.cache_data()
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


@st.cache_resource()
def change_model(current_model, model):
    loaded_model = "None"

    if current_model != model:
        if model == "ECCV16":
            loaded_model = eccv16(pretrained=True).eval()
        elif model == "SIGGRAPH17":
            loaded_model = siggraph17(pretrained=True).eval()
        return loaded_model
    else:
        raise Exception("Model is the same as the current one.")


def format_time(seconds: float) -> str:
    """Formats time in seconds to a human readable format"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{minutes} minutes and {int(seconds)} seconds"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds %= 60
        return f"{hours} hours, {minutes} minutes, and {int(seconds)} seconds"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60
        seconds %= 60
        return f"{days} days, {hours} hours, {minutes} minutes, and {int(seconds)} seconds"


# Function to colorize video frames
def colorize_frame(frame, colorizer) -> np.ndarray:
    tens_l_orig, tens_l_rs = preprocess_img(frame, HW=(256, 256))
    return postprocess_tens(tens_l_orig, colorizer(tens_l_rs).cpu())


def colorize_image(file, loaded_model):
    img = load_img(file)
    # If user input a colored image with 4 channels, discard the fourth channel
    if img.shape[2] == 4:
        img = img[:, :, :3]

    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    out_img = postprocess_tens(tens_l_orig, loaded_model(tens_l_rs).cpu())
    new_img = Image.fromarray((out_img * 255).astype(np.uint8))

    return out_img, new_img