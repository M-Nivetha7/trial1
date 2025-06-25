import os
# 🛠️ Fix for MediaPipe PermissionError
os.environ["MEDIAPIPE_CACHE_DIR"] = os.path.join(os.getcwd(), ".mp_cache")

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import gc

# Initialize MediaPipe Pose (optimized)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=False)
mp_drawing = mp.solutions.drawing_utils

important_body_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.left_angle = 0
        self.right_angle = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            # Calculate angles
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1],
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0]]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * img.shape[1],
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * img.shape[0]]
            self.left_angle = 180 - calculate_angle(left_elbow, left_shoulder,
                                                    [left_shoulder[0], left_shoulder[1] - 100])

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1],
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0]]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * img.shape[1],
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * img.shape[0]]
            self.right_angle = 180 - calculate_angle(right_elbow, right_shoulder,
                                                     [right_shoulder[0], right_shoulder[1] - 100])
        return img

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

def main():
    st.set_page_config(layout="wide", page_title="NeuroTrack Pro | Stroke Therapy Monitoring", page_icon="🧠")

    if "left_angles" not in st.session_state:
        st.session_state.left_angles = []
    if "right_angles" not in st.session_state:
        st.session_state.right_angles = []
    if "timestamps" not in st.session_state:
        st.session_state.timestamps = []
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "last_df" not in st.session_state:
        st.session_state.last_df = None
    if "last_left_meter" not in st.session_state:
        st.session_state.last_left_meter = None
    if "last_right_meter" not in st.session_state:
        st.session_state.last_right_meter = None

    st.title("🧠 NeuroTrack Pro: Stroke Rehab Progress Monitor")
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.subheader("📹 Live Motion Analysis")
        ctx = webrtc_streamer(
            key="stream",
            video_transformer_factory=VideoTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
        )

    with col2:
        st.subheader("🦾 Left Arm")
        meter_left = st.empty()

    with col3:
        st.subheader("💪 Right Arm")
        meter_right = st.empty()

    st.subheader("📈 Progress Graph")
    graph_placeholder = st.empty()
    timer_placeholder = st.empty()

    while ctx.state.playing:
        if ctx.video_transformer:
            left = ctx.video_transformer.left_angle
            right = ctx.video_transformer.right_angle

            st.session_state.left_angles.append(left)
            st.session_state.right_angles.append(right)
            st.session_state.timestamps.append(time.time())

            buf_left = draw_angle_meter(left, "Left Arm")
            buf_right = draw_angle_meter(right, "Right Arm")

            meter_left.image(buf_left)
            meter_right.image(buf_right)

            df = pd.DataFrame({
                "Time": st.session_state.timestamps,
                "Left Arm Angle": st.session_state.left_angles,
                "Right Arm Angle": st.session_state.right_angles
            })
            df["Relative Time"] = df["Time"] - df["Time"].iloc[0]
            graph_placeholder.line_chart(df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])

            st.session_state.last_df = df
            st.session_state.last_left_meter = buf_left
            st.session_state.last_right_meter = buf_right

            elapsed = int(time.time() - st.session_state.start_time)
            mins, secs = divmod(elapsed, 60)
            timer_placeholder.markdown(f"⏳ **Session Time:** {mins:02d}:{secs:02d}")

            # Clean up
            buf_left.close()
            buf_right.close()
            if elapsed % 10 == 0:
                gc.collect()

        time.sleep(0.3)

    if not ctx.state.playing and st.session_state.last_df is not None:
        st.success("✅ Session Complete. Data captured!")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.last_left_meter, caption="Final Left Arm Reading")
        with col2:
            st.image(st.session_state.last_right_meter, caption="Final Right Arm Reading")

        st.subheader("📊 Final Chart")
        st.line_chart(st.session_state.last_df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])

        csv = st.session_state.last_df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv, file_name="neurotrack_session_data.csv", mime="text/csv")

if __name__ == "__main__":
    main()
