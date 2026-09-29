import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import subprocess
import imageio_ffmpeg


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Cricket Batting Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobile-friendly styling
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5%;
    padding-right: 5%;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    h1 {
        font-size: 1.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏏 AI Cricket Batting Analyzer")
st.write("Upload a cricket batting video to analyze body pose, balance, head, shoulder and knee movement.")


# =========================================================
# MEDIAPIPE
# =========================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# =========================================================
# FUNCTIONS
# =========================================================

def angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom == 0:
        return 0

    value = np.dot(ba, bc) / denom
    value = np.clip(value, -1.0, 1.0)

    return np.degrees(np.arccos(value))


def put_text(frame, text, x, y, size=0.55):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# =========================================================
# CONVERT VIDEO TO BROWSER COMPATIBLE MP4
# =========================================================

def convert_video(input_file, output_file):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", input_file,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_file
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.returncode == 0


# =========================================================
# VIDEO ANALYSIS
# =========================================================

def analyze_video(input_path):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Video open nahi ho rahi.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # -----------------------------------------------------
    # Temporary raw video
    # -----------------------------------------------------

    raw_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".avi"
    )

    raw_path = raw_file.name
    raw_file.close()

    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
    )

    writer = cv2.VideoWriter(
        raw_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise Exception("Video writer open nahi hua.")

    # -----------------------------------------------------
    # Analysis variables
    # -----------------------------------------------------

    frame_count = 0
    detected_count = 0

    left_knees = []
    right_knees = []

    shoulder_values = []
    balance_values = []

    head_x = []

    progress = st.progress(0)
    status = st.empty()

    # -----------------------------------------------------
    # Pose
    # -----------------------------------------------------

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:

            success, frame = cap.read()

            if not success:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = pose.process(rgb)

            # =================================================
            # BODY DETECTED
            # =================================================

            if result.pose_landmarks:

                detected_count += 1

                lm = result.pose_landmarks.landmark

                # -------------------------------------------------
                # FULL BODY DOTS + LINES
                # -------------------------------------------------

                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,

                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=3,
                        circle_radius=5
                    ),

                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 0),
                        thickness=3,
                        circle_radius=3
                    )
                )

                # -------------------------------------------------
                # LANDMARKS
                # -------------------------------------------------

                nose = lm[
                    mp_pose.PoseLandmark.NOSE
                ]

                ls = lm[
                    mp_pose.PoseLandmark.LEFT_SHOULDER
                ]

                rs = lm[
                    mp_pose.PoseLandmark.RIGHT_SHOULDER
                ]

                lh = lm[
                    mp_pose.PoseLandmark.LEFT_HIP
                ]

                rh = lm[
                    mp_pose.PoseLandmark.RIGHT_HIP
                ]

                lk = lm[
                    mp_pose.PoseLandmark.LEFT_KNEE
                ]

                rk = lm[
                    mp_pose.PoseLandmark.RIGHT_KNEE
                ]

                la = lm[
                    mp_pose.PoseLandmark.LEFT_ANKLE
                ]

                ra = lm[
                    mp_pose.PoseLandmark.RIGHT_ANKLE
                ]

                # -------------------------------------------------
                # KNEE ANGLES
                # -------------------------------------------------

                left_angle = angle(
                    lh,
                    lk,
                    la
                )

                right_angle = angle(
                    rh,
                    rk,
                    ra
                )

                left_knees.append(
                    left_angle
                )

                right_knees.append(
                    right_angle
                )

                # -------------------------------------------------
                # SHOULDER
                # -------------------------------------------------

                shoulder_difference = abs(
                    ls.y - rs.y
                )

                shoulder_values.append(
                    shoulder_difference
                )

                # -------------------------------------------------
                # HEAD
                # -------------------------------------------------

                head_x.append(
                    nose.x
                )

                # -------------------------------------------------
                # BODY BALANCE
                # -------------------------------------------------

                shoulder_center = (
                    ls.x + rs.x
                ) / 2

                hip_center = (
                    lh.x + rh.x
                ) / 2

                ankle_center = (
                    la.x + ra.x
                ) / 2

                body_center = (
                    shoulder_center +
                    hip_center
                ) / 2

                balance = abs(
                    body_center -
                    ankle_center
                )

                balance_values.append(
                    balance
                )

                # -------------------------------------------------
                # STATUS
                # -------------------------------------------------

                if shoulder_difference < 0.05:
                    shoulder_status = "GOOD"
                else:
                    shoulder_status = "NEED IMPROVEMENT"

                if 80 <= left_angle <= 160:
                    left_status = "GOOD"
                else:
                    left_status = "CHECK"

                if 80 <= right_angle <= 160:
                    right_status = "GOOD"
                else:
                    right_status = "CHECK"

                if balance < 0.08:
                    balance_status = "GOOD"
                else:
                    balance_status = "NEED IMPROVEMENT"

                # -------------------------------------------------
                # HEAD DOT
                # -------------------------------------------------

                hx = int(nose.x * width)
                hy = int(nose.y * height)

                cv2.circle(
                    frame,
                    (hx, hy),
                    10,
                    (0, 255, 255),
                    -1
                )

                # -------------------------------------------------
                # BODY CENTER LINE
                # -------------------------------------------------

                center_pixel = int(
                    body_center * width
                )

                cv2.line(
                    frame,
                    (center_pixel, 0),
                    (center_pixel, height),
                    (0, 255, 255),
                    2
                )

                # -------------------------------------------------
                # INFORMATION BOX
                # -------------------------------------------------

                cv2.rectangle(
                    frame,
                    (10, 10),
                    (470, 230),
                    (0, 0, 0),
                    -1
                )

                put_text(
                    frame,
                    "AI CRICKET ANALYSIS",
                    25,
                    40,
                    0.7
                )

                put_text(
                    frame,
                    f"HEAD: {head_x[-1]:.2f}",
                    25,
                    75
                )

                put_text(
                    frame,
                    f"SHOULDER: {shoulder_status}",
                    25,
                    105
                )

                put_text(
                    frame,
                    f"LEFT KNEE: {left_angle:.0f}  {left_status}",
                    25,
                    135
                )

                put_text(
                    frame,
                    f"RIGHT KNEE: {right_angle:.0f}  {right_status}",
                    25,
                    165
                )

                put_text(
                    frame,
                    f"BALANCE: {balance_status}",
                    25,
                    195
                )

                put_text(
                    frame,
                    "POSE DETECTED",
                    25,
                    220
                )

            else:

                put_text(
                    frame,
                    "BODY NOT DETECTED",
                    20,
                    40,
                    0.7
                )

            # =================================================
            # SAVE FRAME
            # =================================================

            writer.write(frame)

            frame_count += 1

            if total_frames > 0:

                percent = (
                    frame_count /
                    total_frames
                )

                progress.progress(
                    min(percent, 1.0)
                )

                status.write(
                    f"Processing: "
                    f"{frame_count}/{total_frames}"
                )

    cap.release()
    writer.release()

    progress.empty()
    status.empty()

    # =========================================================
    # CONVERT TO H264 MP4
    # =========================================================

    final_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    final_path = final_file.name
    final_file.close()

    success = convert_video(
        raw_path,
        final_path
    )

    try:
        os.remove(raw_path)
    except:
        pass

    if not success:

        try:
            os.remove(final_path)
        except:
            pass

        raise Exception(
            "H.264 video conversion failed."
        )

    # =========================================================
    # RESULTS
    # =========================================================

    if left_knees:
        avg_left = np.mean(left_knees)
    else:
        avg_left = 0

    if right_knees:
        avg_right = np.mean(right_knees)
    else:
        avg_right = 0

    if shoulder_values:
        avg_shoulder = np.mean(
            shoulder_values
        )
    else:
        avg_shoulder = 0

    if balance_values:
        avg_balance = np.mean(
            balance_values
        )
    else:
        avg_balance = 0

    if len(head_x) > 1:
        head_movement = np.std(head_x)
    else:
        head_movement = 0

    detection_rate = (
        detected_count /
        max(frame_count, 1)
    ) * 100

    return {
        "video": final_path,
        "frames": frame_count,
        "detected": detected_count,
        "rate": detection_rate,
        "left_knee": avg_left,
        "right_knee": avg_right,
        "shoulder": avg_shoulder,
        "balance": avg_balance,
        "head": head_movement
    }


# =========================================================
# STREAMLIT UI
# =========================================================

uploaded = st.file_uploader(
    "🎥 Upload Cricket Batting Video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv"
    ]
)

if uploaded is not None:

    st.caption(f"Selected video: {uploaded.name}")
    st.subheader("Original Video")

    st.video(
        uploaded
    )

    if st.button(
        "🏏 ANALYZE BATTING",
        type="primary"
    ):

        # -------------------------------------------------
        # Save input
        # -------------------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded.getbuffer()
        )

        input_file.close()

        try:

            with st.spinner(
                "AI is analyzing the batting video..."
            ):

                result = analyze_video(
                    input_file.name
                )

        except Exception as e:

            st.error(
                "Analysis failed."
            )

            st.code(
                str(e)
            )

            try:
                os.remove(
                    input_file.name
                )
            except:
                pass

            st.stop()

        try:
            os.remove(
                input_file.name
            )
        except:
            pass

        # =================================================
        # SUCCESS
        # =================================================

        st.success(
            "✅ Analysis completed!"
        )

        # =================================================
        # METRICS
        # =================================================

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Frames",
                result["frames"]
            )

        with c2:
            st.metric(
                "Pose Detected",
                result["detected"]
            )

        with c3:
            st.metric(
                "Detection Rate",
                f'{result["rate"]:.1f}%'
            )

        # =================================================
        # ANALYSIS RESULTS
        # =================================================

        st.subheader(
            "📊 Batting Analysis"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.write(
                f'**Left Knee:** '
                f'{result["left_knee"]:.1f}°'
            )

            st.write(
                f'**Right Knee:** '
                f'{result["right_knee"]:.1f}°'
            )

            st.write(
                f'**Shoulder Difference:** '
                f'{result["shoulder"]:.3f}'
            )

        with c2:

            st.write(
                f'**Body Balance:** '
                f'{result["balance"]:.3f}'
            )

            st.write(
                f'**Head Movement:** '
                f'{result["head"]:.3f}'
            )

        # =================================================
        # SUGGESTIONS
        # =================================================

        st.subheader(
            "💡 AI Suggestions"
        )

        if result["head"] < 0.03:
            st.success(
                "Head movement looks controlled."
            )
        else:
            st.warning(
                "Head movement needs improvement."
            )

        if result["shoulder"] < 0.05:
            st.success(
                "Shoulder position looks good."
            )
        else:
            st.warning(
                "Shoulder alignment needs improvement."
            )

        if (
            80 <= result["left_knee"] <= 160
            and
            80 <= result["right_knee"] <= 160
        ):
            st.success(
                "Knee position looks good."
            )
        else:
            st.warning(
                "Knee position needs improvement."
            )

        if result["balance"] < 0.08:
            st.success(
                "Body balance looks good."
            )
        else:
            st.warning(
                "Body balance needs improvement."
            )

        # =================================================
        # ANALYZED VIDEO
        # =================================================

        st.subheader(
            "🎥 AI Analyzed Video"
        )

        if os.path.exists(
            result["video"]
        ):

            with open(
                result["video"],
                "rb"
            ) as f:

                video_data = f.read()

            st.video(
                video_data,
                format="video/mp4"
            )

            st.download_button(
                "⬇️ Download Analyzed Video",
                data=video_data,
                file_name="AI_Cricket_Analysis.mp4",
                mime="video/mp4"
            )

        else:

            st.error(
                "Analyzed video file nahi mili."
            )