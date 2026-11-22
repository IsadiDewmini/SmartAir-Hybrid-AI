import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import hashlib

from pathlib import Path
from ultralytics import YOLO
from smartair_cnn import SmartAirCNN


# ==================================================
# 1. PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="SmartAir | AI Crowd Monitoring",
    page_icon="✈️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

YOLO_PATH = (
    BASE_DIR / "runs" / "detect" / "runs" /
    "smartair" / "airport_model" / "weights" / "best.pt"
)

CNN_PATH = BASE_DIR / "smartair_cnn.pth"
VIDEO_FOLDER = BASE_DIR / "dataset" / "vedios"
OUTPUT_PATH = BASE_DIR / "smartair_dashboard_result.mp4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# 2. DASHBOARD DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #0B1628;
    color: white;
}
h1, h2, h3 {
    color: white;
}
.small-text {
    color: #A9BBD0;
}
.stButton button {
    background-color: #12BFA5;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD AI MODELS
# ==================================================

@st.cache_resource
def load_models():

    yolo_model = YOLO(str(YOLO_PATH))

    cnn_model = SmartAirCNN()

    checkpoint = torch.load(
        CNN_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    cnn_model.load_state_dict(checkpoint)
    cnn_model.to(DEVICE)
    cnn_model.eval()

    return yolo_model, cnn_model


# ==================================================
# 4. CNN PREDICTION
# ==================================================

def predict_cnn(frame, cnn_model):

    resized = cv2.resize(frame, (320, 320))

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    image = rgb.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    image_tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        prediction = cnn_model(image_tensor)

    return max(0.0, float(prediction.item()))


# ==================================================
# 5. CROWD CLASSIFICATION
# ==================================================

def classify_crowd(count, capacity):

    if capacity <= 0:
        return "UNKNOWN", 0.0

    occupancy = (count / capacity) * 100.0

    if occupancy <= 40:
        level = "LOW"

    elif occupancy <= 75:
        level = "MEDIUM"

    else:
        level = "HIGH"

    return level, occupancy


# ==================================================
# 6. VIDEO PROCESSING
# ==================================================

def process_video(video_path, yolo_model, cnn_model, capacity):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open selected video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None

    yolo_counts = []
    cnn_counts = []

    try:

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(OUTPUT_PATH),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError("Could not create output video.")

        st.subheader("Live AI Monitoring")

        live_preview = st.empty()
        progress = st.progress(0)
        status = st.empty()

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO PERSON DETECTION

            results = yolo_model.predict(
                frame,
                conf=0.10,
                iou=0.45,
                device=0 if DEVICE == "cuda" else "cpu",
                verbose=False
            )

            yolo_count = len(results[0].boxes)

            # OWN SMARTAIR CNN CROWD ESTIMATION

            cnn_count = predict_cnn(frame, cnn_model)

            crowd_level, occupancy = classify_crowd(
                cnn_count,
                capacity
            )

            yolo_counts.append(yolo_count)
            cnn_counts.append(cnn_count)

            # DRAW YOLO BOUNDING BOXES

            for box in results[0].boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 150),
                    2
                )

            # ADD AI INFORMATION TO FRAME

            cv2.putText(
                frame,
                f"YOLO Count: {yolo_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 150),
                2
            )

            cv2.putText(
                frame,
                f"CNN Estimate: {cnn_count:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 0),
                2
            )

            cv2.putText(
                frame,
                f"Crowd: {crowd_level} ({occupancy:.1f}%)",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2
            )

            # SAVE PROCESSED FRAME

            writer.write(frame)

            frame_number += 1

            # LIVE PREVIEW WITH YOLO BOXES

            if frame_number % 5 == 0:

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                live_preview.image(
                    rgb_frame,
                    channels="RGB",
                    use_container_width=True
                )

            # PROCESSING PROGRESS

            if total_frames > 0:

                progress.progress(
                    min(frame_number / total_frames, 1.0)
                )

            status.write(
                f"AI Processing: {frame_number} / "
                f"{total_frames} frames"
            )

    finally:

        cap.release()

        if writer is not None:
            writer.release()

    progress.progress(1.0)
    status.success("AI video processing completed!")

    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        raise RuntimeError("Processed video was not created correctly.")

    return yolo_counts, cnn_counts


# ==================================================
# 7. HEADER
# ==================================================

st.markdown(
    "<h1>✈️ SmartAir</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='small-text'>"
    "AI-Powered Airport Crowd Monitoring System"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# 8. SYSTEM OVERVIEW
# ==================================================

st.subheader("System Overview")

col1, col2, col3 = st.columns(3)

with col1:

    if YOLO_PATH.exists():
        st.success("YOLO Model: Available")

    else:
        st.error("YOLO Model: Not Found")


with col2:

    if CNN_PATH.exists():
        st.success("SmartAir CNN: Available")

    else:
        st.error("SmartAir CNN: Not Found")


with col3:

    if DEVICE == "cuda":
        st.success("Processing Device: GPU (CUDA)")

    else:
        st.warning("Processing Device: CPU")

st.divider()


# ==================================================
# 9. CROWD CAPACITY SETTING
# ==================================================

st.subheader("Crowd Capacity Setting")

capacity = st.number_input(
    "Maximum people capacity in the camera area",
    min_value=1,
    max_value=100000,
    value=100,
    step=5
)

st.caption(
    "Set the approximate maximum number of people visible in this camera view. "
    "The same capacity setting will be used for the selected video."
)

st.info(
    "LOW: 0–40% | MEDIUM: above 40–75% | HIGH: above 75% "
    "of the entered capacity."
)

st.divider()


# ==================================================
# 10. VIDEO SOURCE
# ==================================================

st.subheader("Airport Video Monitoring")

video_source = st.radio(
    "Select Video Source",
    ["Dataset Videos", "Upload New Video"],
    horizontal=True
)

selected_video_path = None
uploaded_video = None
video_key = None


# ==================================================
# 11. DATASET VIDEO SELECTION
# ==================================================

if video_source == "Dataset Videos":

    st.write("Select a video from your existing dataset.")

    if not VIDEO_FOLDER.exists():

        st.error(f"Video folder not found: {VIDEO_FOLDER}")

    else:

        video_files = sorted([
            file for file in VIDEO_FOLDER.iterdir()
            if file.is_file()
            and file.suffix.lower() in [
                ".mp4", ".avi", ".mov", ".mkv"
            ]
        ])

        if not video_files:

            st.warning("No videos found in dataset/vedios.")

        else:

            selected_video = st.selectbox(
                "Choose an Airport Video",
                video_files,
                format_func=lambda file: file.name
            )

            selected_video_path = selected_video

            video_key = (
                str(selected_video.resolve())
                + str(selected_video.stat().st_size)
                + str(selected_video.stat().st_mtime)
            )

            st.success(
                f"Selected video: {selected_video.name}"
            )


# ==================================================
# 12. UPLOAD NEW VIDEO
# ==================================================

else:

    uploaded_video = st.file_uploader(
        "Upload an airport CCTV video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        video_bytes = uploaded_video.getvalue()

        video_hash = hashlib.md5(
            video_bytes
        ).hexdigest()

        video_key = uploaded_video.name + video_hash

        st.success("Video uploaded successfully!")


# ==================================================
# 13. AUTOMATIC AI ANALYSIS
# ==================================================

if video_key is not None:

    st.divider()
    st.subheader("Original Airport Video")

    if selected_video_path is not None:

        st.video(str(selected_video_path))

    elif uploaded_video is not None:

        st.video(uploaded_video)

    if not YOLO_PATH.exists():

        st.error("YOLO model file not found.")

    elif not CNN_PATH.exists():

        st.error("SmartAir CNN model file not found.")

    else:

        # Capacity is part of the key so changing it triggers a new analysis.

        analysis_key = f"{video_key}_capacity_{capacity}"

        if st.session_state.get("analysis_key") != analysis_key:

            try:

                with st.spinner("Loading AI models..."):

                    yolo_model, cnn_model = load_models()

                st.success("Both AI models loaded successfully!")

                # PREPARE INPUT VIDEO

                if selected_video_path is not None:

                    input_path = selected_video_path

                    yolo_counts, cnn_counts = process_video(
                        input_path,
                        yolo_model,
                        cnn_model,
                        capacity
                    )

                else:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_video.name).suffix
                    ) as temp_file:

                        temp_file.write(
                            uploaded_video.getbuffer()
                        )

                        input_path = Path(temp_file.name)

                    try:

                        yolo_counts, cnn_counts = process_video(
                            input_path,
                            yolo_model,
                            cnn_model,
                            capacity
                        )

                    finally:

                        input_path.unlink(missing_ok=True)

                st.session_state.analysis_key = analysis_key

                st.session_state.analysis_result = {
                    "yolo_counts": yolo_counts,
                    "cnn_counts": cnn_counts
                }

            except Exception as error:

                st.error(f"Processing error: {error}")

        # DISPLAY RESULTS

        if st.session_state.get("analysis_key") == analysis_key:

            result = st.session_state.analysis_result

            yolo_counts = result["yolo_counts"]
            cnn_counts = result["cnn_counts"]

            if yolo_counts and cnn_counts:

                avg_yolo = sum(yolo_counts) / len(yolo_counts)
                avg_cnn = sum(cnn_counts) / len(cnn_counts)

                average_level, average_occupancy = classify_crowd(
                    avg_cnn,
                    capacity
                )

                st.divider()
                st.subheader("AI Analysis Results")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Average YOLO Count",
                        f"{avg_yolo:.2f}"
                    )

                with col2:

                    st.metric(
                        "Average CNN Estimate",
                        f"{avg_cnn:.2f}"
                    )

                with col3:

                    st.metric(
                        "Frames Processed",
                        len(yolo_counts)
                    )

                st.subheader("Crowd Capacity Analysis")

                col4, col5, col6 = st.columns(3)

                with col4:

                    st.metric(
                        "Area Capacity",
                        f"{capacity} people"
                    )

                with col5:

                    st.metric(
                        "Estimated Occupancy",
                        f"{average_occupancy:.1f}%"
                    )

                with col6:

                    st.metric(
                        "Crowd Level",
                        average_level
                    )

                st.caption(
                    "Crowd level is estimated using the SmartAir CNN count "
                    "compared with the capacity entered above."
                )

                st.divider()
                st.subheader("YOLO vs CNN Comparison")

                st.write(
                    f"YOLO average detection: **{avg_yolo:.2f} people**"
                )

                st.write(
                    f"CNN average estimation: **{avg_cnn:.2f} people**"
                )

                st.divider()

                # PROCESSED VIDEO PLAYBACK

                st.subheader("Processed Airport Video")

                if OUTPUT_PATH.exists():

                    video_size = OUTPUT_PATH.stat().st_size

                    st.write(
                        f"Processed video size: "
                        f"{video_size / (1024 * 1024):.2f} MB"
                    )

                    try:

                        with open(OUTPUT_PATH, "rb") as video_file:

                            video_bytes = video_file.read()

                        st.video(
                            video_bytes,
                            format="video/mp4"
                        )

                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name="SmartAir_Processed_Video.mp4",
                            mime="video/mp4"
                        )

                    except Exception as error:

                        st.error(
                            f"Could not display processed video: {error}"
                        )

                else:

                    st.error("Processed video file not found.")

                st.success("AI analysis completed successfully!")

            else:

                st.warning("No video frames were processed.")


# ==================================================
# 14. FOOTER
# ==================================================

st.divider()

st.markdown(
    "<p style='text-align:center;color:#8298B2;'>"
    "SmartAir AI Crowd Monitoring | University Project"
    "</p>",
    unsafe_allow_html=True
)
