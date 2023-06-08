import streamlit as st
from streamlit_lottie import st_lottie

from models.deep_colorization.colorizers import eccv16
from utils import load_lottieurl, change_model

st.set_page_config(page_title="Image & Video Colorizer", page_icon="🎨", layout="wide")


loaded_model = eccv16(pretrained=True).eval()
current_model = "None"


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
        st.info('This feature hasn\'t been implemented yet', icon="ℹ️")


if __name__ == "__main__":
    main()
    st.markdown(
        "###### Made with :heart: by [Clément Delteil](https://www.linkedin.com/in/clementdelteil/) [![this is an "
        "image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/clementdelteil)")
