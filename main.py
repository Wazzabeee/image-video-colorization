import streamlit as st
from models.deep_colorization.colorizers import *
import cv2
from PIL import Image
import pathlib
import tempfile
import moviepy.editor as mp
import time
from tqdm import tqdm


def format_time(seconds):
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
def colorize_frame(frame, colorizer):
    (tens_l_orig, tens_l_rs) = preprocess_img(frame, HW=(256, 256))
    return postprocess_tens(tens_l_orig, colorizer(tens_l_rs).cpu())


image = Image.open(r'img/streamlit.png')  # Brand logo image (optional)

APP_DIR = pathlib.Path(__file__).parent.absolute()

LOCAL_DIR = APP_DIR / "local_video"
LOCAL_DIR.mkdir(exist_ok=True)
save_dir = LOCAL_DIR / "output"
save_dir.mkdir(exist_ok=True)

print(APP_DIR)
print(LOCAL_DIR)
print(save_dir)

# Create two columns with different width
col1, col2 = st.columns([0.8, 0.2])
with col1:  # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF4B4B;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your photo or video here...</p>', unsafe_allow_html=True)

with col2:  # To display brand logo
    st.image(image, width=100)

# Add a header and expander in side bar
st.sidebar.markdown('<p class="font">Color Revive App</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
    st.write("""
        Use this simple app to colorize your black and white images and videos with state of the art models.
     """)

# Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg', 'mp4'])

# Add 'before' and 'after' columns
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[1].lower()

    if file_extension in ['jpg', 'png', 'jpeg']:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
            st.image(image, width=300)

        # Add conditional statements to take the user input values
        with col2:
            st.markdown('<p style="text-align: center;">After</p>', unsafe_allow_html=True)
            filter = st.sidebar.radio('Colorize your image with:',
                                      ['Original', 'ECCV 16', 'SIGGRAPH 17'])
            if filter == 'ECCV 16':
                colorizer_eccv16 = eccv16(pretrained=True).eval()
                img = load_img(uploaded_file)
                (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
                out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
                st.image(out_img_eccv16, width=300)
            elif filter == 'SIGGRAPH 17':
                colorizer_siggraph17 = siggraph17(pretrained=True).eval()
                img = load_img(uploaded_file)
                (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
                out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
                st.image(out_img_siggraph17, width=300)
            else:
                st.image(image, width=300)
    elif file_extension == 'mp4':  # If uploaded file is a video
        # Save the video file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        # Open the video using cv2.VideoCapture
        video = cv2.VideoCapture(temp_file.name)

        # Get video information
        fps = video.get(cv2.CAP_PROP_FPS)

        # Create two columns for video display
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
            st.video(temp_file.name)

        with col2:
            st.markdown('<p style="text-align: center;">After</p>', unsafe_allow_html=True)
            filter = st.sidebar.radio('Colorize your video with:',
                                      ['Original', 'ECCV 16', 'SIGGRAPH 17'])
            if filter == 'ECCV 16':
                colorizer = eccv16(pretrained=True).eval()
            elif filter == 'SIGGRAPH 17':
                colorizer = siggraph17(pretrained=True).eval()

            if filter != 'Original':
                with st.spinner("Colorizing frames..."):
                    # Colorize video frames and store in a list
                    output_frames = []
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.empty()

                    start_time = time.time()
                    for i in tqdm(range(total_frames), unit='frame', desc="Progress"):
                        ret, frame = video.read()
                        if not ret:
                            break

                        colorized_frame = colorize_frame(frame, colorizer)
                        output_frames.append((colorized_frame * 255).astype(np.uint8))

                        elapsed_time = time.time() - start_time
                        frames_completed = len(output_frames)
                        frames_remaining = total_frames - frames_completed
                        time_remaining = (frames_remaining / frames_completed) * elapsed_time

                        progress_bar.progress(frames_completed / total_frames)

                        if frames_completed < total_frames:
                            progress_bar.text(f"Time Remaining: {format_time(time_remaining)}")
                        else:
                            progress_bar.empty()

                with st.spinner("Merging frames to video..."):
                    print("finished")
                    frame_size = output_frames[0].shape[:2]
                    print(frame_size)
                    output_filename = "output.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 video
                    print(fps)
                    out = cv2.VideoWriter(output_filename, fourcc, fps, (3840, 2160))

                    # Display the colorized video using st.video
                    for frame in output_frames:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        out.write(frame_bgr)

                    out.release()

                    # Convert the output video to a format compatible with Streamlit
                    converted_filename = "converted_output.mp4"
                    clip = mp.VideoFileClip(output_filename)
                    clip.write_videofile(converted_filename, codec="libx264")

                    # Display the converted video using st.video()
                    st.video(converted_filename)

                    # Close and delete the temporary file after processing
                    video.release()
                    temp_file.close()

# Add a feedback section in the sidebar
st.sidebar.title(' ')  # Used to create some space between the filter widget and the comments section
st.sidebar.markdown(' ')  # Used to create some space between the filter widget and the comments section
st.sidebar.subheader('Please help us improve!')
with st.sidebar.form(key='columns_in_form',
                     clear_on_submit=True):  # set clear_on_submit=True so that the form will be reset/cleared once
    # it's submitted
    rating = st.slider("Please rate the app", min_value=1, max_value=5, value=3,
                       help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
    text = st.text_input(label='Please leave your feedback here')
    submitted = st.form_submit_button('Submit')
    if submitted:
        st.write('Thanks for your feedback!')
        st.markdown('Your Rating:')
        st.markdown(rating)
        st.markdown('Your Feedback:')
        st.markdown(text)
