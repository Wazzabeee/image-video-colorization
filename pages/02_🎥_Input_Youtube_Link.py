import time

import cv2
import moviepy.editor as mp
import numpy as np
import streamlit as st
from pytube import YouTube
from tqdm import tqdm


from utils import format_time, colorize_frame, change_model, load_model, setup_columns, set_page_config

set_page_config()
loaded_model = load_model()
col2 = setup_columns()
current_model = None

with col2:
    st.write(
        """
    ## B&W Videos Colorizer
    ##### Input a YouTube black and white video link and get a colorized version of it.
    ###### ➠ This space is using CPU Basic so it might take a while to colorize a video.
    ###### ➠ If you want more models and GPU available please support this space by donating."""
    )


@st.cache_data()
def download_video(link):
    """
    Download video from YouTube
    """
    yt = YouTube(link)
    video = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
        .download(filename="video.mp4")
    )
    return video


def main():
    """
    Main function
    """
    model = st.selectbox(
        "Select Model (Both models have their pros and cons,"
        "I recommend trying both and keeping the best for you task)",
        ["ECCV16", "SIGGRAPH17"],
        index=0,
    )

    loaded_model = change_model(current_model, model)
    st.write(f"Model is now {model}")

    link = st.text_input("YouTube Link (The longer the video, the longer the processing time)")
    if st.button("Colorize"):
        yt_video = download_video(link)
        print(yt_video)
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
            st.video(yt_video)
        with col2:
            st.markdown('<p style="text-align: center;">After</p>', unsafe_allow_html=True)
            with st.spinner("Colorizing frames..."):
                # Colorize video frames and store in a list
                output_frames = []

                audio = mp.AudioFileClip("video.mp4")
                video = cv2.VideoCapture("video.mp4")

                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video.get(cv2.CAP_PROP_FPS)

                progress_bar = st.progress(0)  # Create a progress bar
                start_time = time.time()
                time_text = st.text("Time Remaining: ")  # Initialize text value

                for _ in tqdm(range(total_frames), unit="frame", desc="Progress"):
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
                    file_name="colorized_video.mp4",
                )

                # Close and delete the temporary file after processing
                video.release()


if __name__ == "__main__":
    main()
    st.markdown(
        "###### Made with :heart: by [Clément Delteil](https://www.linkedin.com/in/clementdelteil/) [![this is an "
        "image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/clementdelteil)"
    )
