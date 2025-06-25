import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import time
from io import BytesIO
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# MediaPipe Tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark

# Set up model path
MODEL_PATH = "models/pose_landmark_heavy.tflite"

# Define which joints to use
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def draw_angle_meter(angle, label):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    circle = plt.Circle((0, 0), 1, color=(0.2, 0.2, 0.2), fill=True)
    ax.add_artist(circle)

    if angle > 120:
        arc_color = "#4CAF50"
        level = "Excellent"
    elif 60 < angle <= 120:
        arc_color = "#FFC107"
        level = "Moderate"
    else:
        arc_color = "#F44336"
        level = "Needs Work"

    theta = np.linspace(np.pi, np.pi - (np.pi * (angle / 180)), 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color=arc_color, linewidth=8)

    ax.text(0, 0, f"{int(angle)}°", ha='center', va='center', fontsize=20, color='white', fontweight='bold')
    ax.text(0, -1.3, label, ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    ax.text(0, 1.2, level, ha='center', va='center', fontsize=14, color=arc_color, fontweight='bold')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', transparent=True, dpi=100)
    buf.seek(0)
    return buf

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.left_angle = 0
        self.right_angle = 0

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
        self.landmarker = PoseLandmarker.create_from_options(options)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)

        if results.pose_landmarks:
            h, w, _ = rgb.shape
            landmarks = results.pose_landmarks[0]

            def get_xy(landmark: NormalizedLandmark):
                return [landmark.x * w, landmark.y * h]

            try:
                ls = get_xy(landmarks[11])  # Left Shoulder
                le = get_xy(landmarks[13])  # Left Elbow
                lh = get_xy(landmarks[15])  # Left Wrist

                rs = get_xy(landmarks[12])  # Right Shoulder
                re = get_xy(landmarks[14])  # Right Elbow
                rh = get_xy(landmarks[16])  # Right Wrist

                self.left_angle = calculate_angle(lh, le, ls)
                self.right_angle = calculate_angle(rh, re, rs)

            except:
                pass

        return img


def main():
    st.set_page_config(page_title="NeuroTrack Cloud", layout="wide")

    st.title("🧠 NeuroTrack Pro (Streamlit Cloud)")
    st.write("Pose detection using manually bundled MediaPipe model")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader("📹 Live Feed")
        ctx = webrtc_streamer(
            key="pose-tracker",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
        )

    with col2:
        st.subheader("🦾 Left Arm")
        meter_left = st.empty()
    with col3:
        st.subheader("💪 Right Arm")
        meter_right = st.empty()

    # Data collectors
    if "data" not in st.session_state:
        st.session_state.data = []

    graph_placeholder = st.empty()
    timer_placeholder = st.empty()

    start_time = time.time()
    while ctx.state.playing:
        vp = ctx.video_processor
        if vp:
            left, right = vp.left_angle, vp.right_angle
            now = time.time()
            elapsed = now - start_time

            st.session_state.data.append((elapsed, left, right))

            buf_l = draw_angle_meter(left, "Left Arm")
            buf_r = draw_angle_meter(right, "Right Arm")
            meter_left.image(buf_l)
            meter_right.image(buf_r)

            df = pd.DataFrame(st.session_state.data, columns=["Time", "Left", "Right"])
            graph_placeholder.line_chart(df.set_index("Time"))

            mins, secs = divmod(int(elapsed), 60)
            timer_placeholder.markdown(f"⏳ **Session Time:** {mins:02}:{secs:02}")

        time.sleep(0.3)

    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data, columns=["Time", "Left", "Right"])
        st.success("✅ Session Complete. Download your data below.")
        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="pose_data.csv", mime="text/csv")

if __name__ == "__main__":
    main()
