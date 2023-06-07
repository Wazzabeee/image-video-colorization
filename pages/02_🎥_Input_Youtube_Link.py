import requests
import streamlit as st
from streamlit_lottie import st_lottie

from models.deep_colorization.colorizers import eccv16, siggraph17

st.set_page_config(page_title="Image & Video Colorizer", page_icon="🎨", layout="wide")


# Define a function that we can use to load lottie files from a link.
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


loaded_model = eccv16(pretrained=True).eval()
current_model = "None"


def change_model(current_model, model):
    if current_model != model:
        if model == "ECCV16":
            loaded_model = eccv16(pretrained=True).eval()
        elif model == "SIGGRAPH17":
            loaded_model = siggraph17(pretrained=True).eval()
        return loaded_model
    else:
        raise Exception("Model is the same as the current one.")


col1, col2 = st.columns([1, 3])
with col1:
    lottie = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_RHdEuzVfEL.json")
    st_lottie(lottie)

with col2:
    st.write("""
    ## B&W Videos Colorizer
    ##### Input a YouTube black and white video link and get a colorized version of it.
    ###### I recommend starting with the first model and then experimenting with the second one.""")


def main():
    model = st.selectbox(
        "Select Model (Both models have their pros and cons, I recommend to try both and keep the best for you task)",
        ["ECCV16", "SIGGRAPH17"], index=0)

    loaded_model = change_model(current_model, model)
    st.write(f"Model is now {model}")

    link = st.text_input("YouTube Link (The longer the video, the longer the processing time)")
    if st.button("Colorize"):
        print("yo")


if __name__ == "__main__":
    main()
    st.markdown(
        "###### Made with :heart: by [Clément Delteil](https://www.linkedin.com/in/clementdelteil/) [![this is an "
        "image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/clementdelteil)")
