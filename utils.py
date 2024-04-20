import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie

from models.deep_colorization import eccv16
from models.deep_colorization import siggraph17
from models.deep_colorization import postprocess_tens, preprocess_img, load_img


class SameModelException(ValueError):
    """Exception raised when the same model is attempted to be reloaded."""


def set_page_config():
    """
    Sets up the page config.
    """
    st.set_page_config(page_title="Image & Video Colorizer", page_icon="🎨", layout="wide")


def load_model():
    """
    Loads the default model.
    """
    return eccv16(pretrained=True).eval()


def setup_columns():
    """
    Sets up the columns.
    """
    col1, col2 = st.columns([1, 3])
    lottie = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_RHdEuzVfEL.json")
    with col1:
        st_lottie(lottie)
    return col2


# Define a function that we can use to load lottie files from a link.
@st.cache_data()
def load_lottieurl(url: str):
    """
    Load lottieurl image
    """
    try:
        r = requests.get(url, timeout=10)  # Timeout set to 10 seconds
        r.raise_for_status()  # This will raise an exception for HTTP errors
        return r.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


@st.cache_resource()
def change_model(current_model, model):
    """
    Change model
    """
    loaded_model = "None"

    if current_model != model:
        if model == "ECCV16":
            loaded_model = eccv16(pretrained=True).eval()
        elif model == "SIGGRAPH17":
            loaded_model = siggraph17(pretrained=True).eval()
        return loaded_model

    raise SameModelException("Model is the same as the current one.")


def format_time(seconds: float) -> str:
    """Formats time in seconds to a human readable format"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    if seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{minutes} minutes and {int(seconds)} seconds"
    if seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds %= 60
        return f"{hours} hours, {minutes} minutes, and {int(seconds)} seconds"

    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    seconds %= 60
    return f"{days} days, {hours} hours, {minutes} minutes, and {int(seconds)} seconds"


# Function to colorize video frames
def colorize_frame(frame, colorizer) -> np.ndarray:
    """
    Colorize frame
    """
    tens_l_orig, tens_l_rs = preprocess_img(frame, HW=(256, 256))
    return postprocess_tens(tens_l_orig, colorizer(tens_l_rs).cpu())


def colorize_image(file, loaded_model):
    """
    Colorize image
    """
    img = load_img(file)
    # If user input a colored image with 4 channels, discard the fourth channel
    if img.shape[2] == 4:
        img = img[:, :, :3]

    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    out_img = postprocess_tens(tens_l_orig, loaded_model(tens_l_rs).cpu())
    new_img = Image.fromarray((out_img * 255).astype(np.uint8))

    return out_img, new_img
