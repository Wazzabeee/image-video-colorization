import os
import zipfile

import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie

from models.deep_colorization.colorizers import eccv16
from utils import colorize_image, change_model, load_lottieurl

st.set_page_config(page_title="Image & Video Colorizer", page_icon="🎨", layout="wide")


loaded_model = eccv16(pretrained=True).eval()
current_model = "None"


col1, col2 = st.columns([1, 3])
with col1:
    lottie = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_RHdEuzVfEL.json")
    st_lottie(lottie)

with col2:
    st.write("""
    ## B&W Images Colorizer
    ##### Input a black and white image and get a colorized version of it.
    ###### ➠ If you want to colorize multiple images just upload them all at once.
    ###### ➠ Uploading already colored images won't raise errors but images won't look good.
    ###### ➠ I recommend starting with the first model and then experimenting with the second one.""")


def main():
    model = st.selectbox(
        "Select Model (Both models have their pros and cons, I recommend to try both and keep the best for you task)",
        ["ECCV16", "SIGGRAPH17"], index=0)

    # Make the user select a model
    loaded_model = change_model(current_model, model)
    st.write(f"Model is now {model}")

    # Ask the user if he wants to see colorization
    display_results = st.checkbox('Display results in real time', value=True)

    # Input for the user to upload images
    uploaded_file = st.file_uploader("Upload your photos here...", type=['jpg', 'png', 'jpeg'],
                                     accept_multiple_files=True)

    # If the user clicks on the button
    if st.button("Colorize"):
        # If the user uploaded images
        if uploaded_file is not None:
            if display_results:
                col1, col2 = st.columns([0.5, 0.5])
                with col1:
                    st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<p style="text-align: center;">After</p>', unsafe_allow_html=True)
            else:
                col1, col2, col3 = st.columns(3)

            for i, file in enumerate(uploaded_file):
                file_extension = os.path.splitext(file.name)[1].lower()
                if file_extension in ['.jpg', '.png', '.jpeg']:
                    image = Image.open(file)
                    if display_results:
                        with col1:
                            st.image(image, use_column_width="always")
                        with col2:
                            with st.spinner("Colorizing image..."):
                                out_img, new_img = colorize_image(file, loaded_model)
                                new_img.save("IMG_" + str(i + 1) + ".jpg")
                                st.image(out_img, use_column_width="always")

                    else:
                        out_img, new_img = colorize_image(file, loaded_model)
                        new_img.save("IMG_" + str(i + 1) + ".jpg")

            if len(uploaded_file) > 1:
                # Create a zip file
                zip_filename = "colorized_images.zip"
                with zipfile.ZipFile(zip_filename, "w") as zip_file:
                    # Add colorized images to the zip file
                    for i in range(len(uploaded_file)):
                        zip_file.write("IMG_" + str(i + 1) + ".jpg", "IMG_" + str(i) + ".jpg")
                with col2:
                    # Provide the zip file data for download
                    st.download_button(
                        label="Download Colorized Images" if len(uploaded_file) > 1 else "Download Colorized Image",
                        data=open(zip_filename, "rb").read(),
                        file_name=zip_filename,
                    )
            else:
                with col2:
                    st.download_button(
                        label="Download Colorized Image",
                        data=open("IMG_1.jpg", "rb").read(),
                        file_name="IMG_1.jpg",
                    )

        else:
            st.warning('Upload a file', icon="⚠️")


if __name__ == "__main__":
    main()
    st.markdown(
        "###### Made with :heart: by [Clément Delteil](https://www.linkedin.com/in/clementdelteil/) [![this is an "
        "image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/clementdelteil)")
