import os
import tempfile
import time

import cv2
import moviepy.editor as mp
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
from tqdm import tqdm

from models.deep_colorization.colorizers import eccv16
from utils import load_lottieurl, format_time, colorize_frame, change_model

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
    ##### Upload a black and white video and get a colorized version of it.
    ###### ➠ This space is using CPU Basic so it might take a while to colorize a video.
    ###### ➠ If you want more models and GPU available please support this space by donating.""")


def main():
    model = st.selectbox(
        "Select Model (Both models have their pros and cons, I recommend to try both and keep the best for your task)",
        ["ECCV16", "SIGGRAPH17"], index=0)

    loaded_model = change_model(current_model, model)
    st.write(f"Model is now {model}")

    uploaded_file = st.file_uploader("Upload your video here...", type=['mp4', 'mov', 'avi', 'mkv'])

    if st.button("Colorize"):
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                # Save the video file to a temporary location
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())

                audio = mp.AudioFileClip(temp_file.name)

                # Open the video using cv2.VideoCapture
                video = cv2.VideoCapture(temp_file.name)

                # Get video information
                fps = video.get(cv2.CAP_PROP_FPS)

                col1, col2 = st.columns([0.5, 0.5])
                with col1:
                    st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
                    st.video(temp_file.name)

                with col2:
                    st.markdown('<p style="text-align: center;">After</p>', unsafe_allow_html=True)

                    with st.spinner("Colorizing frames..."):
                        # Colorize video frames and store in a list
                        output_frames = []
                        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        progress_bar = st.progress(0)  # Create a progress bar

                        start_time = time.time()
                        time_text = st.text("Time Remaining: ")  # Initialize text value

                        for _ in tqdm(range(total_frames), unit='frame', desc="Progress"):
                            ret, frame = video.read()
                            if not ret:
                                break

                            colorized_frame = colorize_frame(frame, loaded_model)
                            output_frames.append((colorized_frame * 255).astype(np.uint8))

                            elapsed_time = time.time() - start_time
                            frames_completed = len(output_frames)
                            frames_remaining = total_frames - frames_completed
                            time_remaining = (frames_remaining / frames_completed) * elapsed_time

                            progress_bar.progress(frames_completed / total_frames)  # Update progress bar

                            if frames_completed < total_frames:
                                time_text.text(f"Time Remaining: {format_time(time_remaining)}")  # Update text value
                            else:
                                time_text.empty()  # Remove text value
                                progress_bar.empty()

                    with st.spinner("Merging frames to video..."):
                        frame_size = output_frames[0].shape[:2]
                        output_filename = "output.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 video
                        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_size[1], frame_size[0]))

                        # Display the colorized video using st.video
                        for frame in output_frames:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                            out.write(frame_bgr)

                        out.release()

                        # Convert the output video to a format compatible with Streamlit
                        converted_filename = "converted_output.mp4"
                        clip = mp.VideoFileClip(output_filename)
                        clip = clip.set_audio(audio)

                        clip.write_videofile(converted_filename, codec="libx264")

                        # Display the converted video using st.video()
                        st.video(converted_filename)
                        st.balloons()

                        # Add a download button for the colorized video
                        st.download_button(
                            label="Download Colorized Video",
                            data=open(converted_filename, "rb").read(),
                            file_name="colorized_video.mp4"
                        )

                        # Close and delete the temporary file after processing
                        video.release()
                        temp_file.close()


if __name__ == "__main__":
    main()
    st.markdown(
        "###### Made with :heart: by [Clément Delteil](https://www.linkedin.com/in/clementdelteil/) [![this is an "
        "image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/clementdelteil)")
