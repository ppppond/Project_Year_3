import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageEnhance
import pandas as pd
import os
import cv2
import tempfile
import yt_dlp
from ultralytics import YOLO

# ==========================================
# 1. CONFIG & SETUP
# ==========================================
st.set_page_config(
    page_title="Img Detection",
    layout="wide",
    page_icon="🧊",
)

BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
IMG_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def create_yaml(class_list):
    yaml_content = f"""
path: {DATASET_DIR}
train: images
val: images
names:
"""
    for idx, name in enumerate(class_list):
        yaml_content += f"  {idx}: {name}\n"

    with open(os.path.join(DATASET_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

def transform_image(img, rotate, flip_h, flip_v, bright, cont, col):
    if rotate != 0:
        img = img.rotate(rotate, expand=True)
    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    img = ImageEnhance.Brightness(img).enhance(bright)
    img = ImageEnhance.Contrast(img).enhance(cont)
    img = ImageEnhance.Color(img).enhance(col)
    return img

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📸 Img Detection")
menu = st.sidebar.radio(
    "เมนูใช้งาน",
    [
        "1. Labeling (วาดกรอบ)",
        "2. Train (สอน AI)",
        "3. Predict (ทำนาย)",
        "4. Webcam (กล้องสด)",
        "5. Video Scan (วิดีโอ)",
    ],
)

# ==========================================
# 1. LABELING (MULTI IMAGE)
# ==========================================
if "Labeling" in menu:
    st.header("🖌️ 1. Labeling: Multi-Image Annotation")

    col1, col2 = st.columns([1, 3])

    # ---------- LEFT ----------
    with col1:
        st.markdown("### ⚙️ ตั้งค่า Class")

        if "class_list" not in st.session_state:
            st.session_state.class_list = ["dog", "cat", "person"]

        class_text = st.text_area(
            "รายชื่อวัตถุ (บรรทัดละชื่อ)",
            "\n".join(st.session_state.class_list),
            height=120,
        )
        st.session_state.class_list = [
            c.strip() for c in class_text.split("\n") if c.strip()
        ]

        selected_class = st.selectbox(
            "👉 กำลังจะวาด:",
            st.session_state.class_list,
        )

        stroke_width = st.slider("ความหนาเส้น", 1, 5, 2)

        st.divider()
        st.markdown("### 🖼️ Image Config")

        rotate_angle = st.slider("🔄 Rotate", -180, 180, 0, step=5)
        flip_h = st.checkbox("↔️ Flip Horizontal")
        flip_v = st.checkbox("↕️ Flip Vertical")
        brightness = st.slider("☀️ Brightness", 0.2, 2.0, 1.0)
        contrast = st.slider("🎚️ Contrast", 0.2, 2.0, 1.0)
        color = st.slider("🎨 Color", 0.2, 2.0, 1.0)

    # ---------- RIGHT ----------
    with col2:
        uploaded_files = st.file_uploader(
            "📂 อัปโหลดรูปภาพ (หลายรูปได้)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            file_names = [f.name for f in uploaded_files]

            selected_idx = st.selectbox(
                "🖼️ เลือกรูปที่จะ Label",
                range(len(file_names)),
                format_func=lambda i: file_names[i],
            )

            uploaded_file = uploaded_files[selected_idx]

            orig_image = Image.open(uploaded_file).convert("RGB")
            image = transform_image(
                orig_image,
                rotate_angle,
                flip_h,
                flip_v,
                brightness,
                contrast,
                color,
            )

            w, h = image.size

            canvas_key = f"canvas_{uploaded_file.name}_{rotate_angle}_{flip_h}_{flip_v}_{brightness}_{contrast}_{color}"

            canvas = st_canvas(
                fill_color="rgba(0, 150, 255, 0.2)",
                stroke_width=stroke_width,
                stroke_color="#0066CC",
                background_image=image,
                update_streamlit=True,
                height=500,
                drawing_mode="rect",
                key=canvas_key,
            )

            if canvas.json_data is not None:
                objects = pd.json_normalize(canvas.json_data["objects"])
                if not objects.empty:
                    st.success(f"✅ พบ {len(objects)} กรอบ")

                    if st.button("💾 บันทึกข้อมูล (Save)", type="primary"):
                        suffix = []
                        if rotate_angle != 0:
                            suffix.append(f"rot{rotate_angle}")
                        if flip_h:
                            suffix.append("fh")
                        if flip_v:
                            suffix.append("fv")
                        if brightness != 1.0:
                            suffix.append(f"b{brightness}")
                        if contrast != 1.0:
                            suffix.append(f"c{contrast}")
                        if color != 1.0:
                            suffix.append(f"col{color}")
                        suffix = "_".join(suffix) if suffix else "orig"

                        img_name = f"{os.path.splitext(uploaded_file.name)[0]}_{suffix}.jpg"
                        label_name = img_name.replace(".jpg", ".txt")

                        image.save(os.path.join(IMG_DIR, img_name))

                        lines = []
                        for _, row in objects.iterrows():
                            xc = (row["left"] + row["width"] / 2) / w
                            yc = (row["top"] + row["height"] / 2) / h
                            nw = row["width"] / w
                            nh = row["height"] / h
                            cid = st.session_state.class_list.index(selected_class)
                            lines.append(
                                f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
                            )

                        with open(os.path.join(LABEL_DIR, label_name), "w") as f:
                            f.write("\n".join(lines))

                        st.toast("บันทึกเรียบร้อย 🎉")

# ==========================================
# 2. TRAIN
# ==========================================
elif "Train" in menu:
    st.header("🚀 2. Train Model")

    class_text = st.text_area(
        "Class List",
        "\n".join(st.session_state.get("class_list", [])),
        height=100,
    )
    classes = [c.strip() for c in class_text.split("\n") if c.strip()]

    if st.button("📄 สร้าง data.yaml"):
        create_yaml(classes)
        st.success("สร้าง data.yaml สำเร็จ")

    epochs = st.number_input("Epochs", 10, 300, 30, step=10)
    batch = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)

    if st.button("🚀 Start Training"):
        model = YOLO("yolov8n.pt")
        model.train(
            data=os.path.join(DATASET_DIR, "data.yaml"),
            epochs=epochs,
            batch=batch,
            imgsz=640,
            project="runs/detect",
            name="my_custom_model",
            exist_ok=True,
        )
        st.success("🎉 Train เสร็จแล้ว")

# ==========================================
# 3. PREDICT (แก้ไข: สี + Confidence Slider)
# ==========================================
elif "Predict" in menu:
    st.header("🔮 3. Predict")

    col1, col2 = st.columns(2)
    
    with col1:
        model_path = st.text_input(
            "📂 Path Model",
            "runs/detect/my_custom_model/weights/best.pt"
        )
        # เพิ่ม Slider ปรับความมั่นใจ (ช่วยให้เจอของง่ายขึ้นถ้าโมเดลยังไม่เก่ง)
        conf_threshold = st.slider("🎚️ ความมั่นใจ (Confidence)", 0.0, 1.0, 0.25, 0.05)

    with col2:
        img_file = st.file_uploader("🖼️ เลือกรูปภาพ", type=["jpg", "png", "jpeg"])

    if img_file and st.button("🔍 เริ่มทำนาย (Predict)", type="primary"):
        # 1. โหลดโมเดล (ใส่ Try-Catch กัน Error)
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
            st.stop()

        # 2. แปลงรูปและส่งเข้าโมเดล
        pil_img = Image.open(img_file)
        
        # ใส่ค่า conf เพื่อกรองความมั่นใจ
        results = model(pil_img, conf=conf_threshold)

        # 3. ดึงรูปผลลัพธ์ (ซึ่งเป็น BGR)
        res_plotted_bgr = results[0].plot()

        # 4. ⭐ แก้สีจาก BGR -> RGB ตรงนี้ ⭐
        res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)

        # 5. แสดงผล
        st.image(res_plotted_rgb, caption="ผลลัพธ์การทำนาย", use_column_width=True)
        
        # แสดงจำนวนที่เจอ
        boxes = results[0].boxes
        if len(boxes) > 0:
            st.success(f"✅ เจอวัตถุทั้งหมด {len(boxes)} ชิ้น")
        else:
            st.warning("⚠️ ไม่เจอวัตถุ (ลองลดค่า Confidence ลงดูนะครับ)")

# ==========================================
# 4. WEBCAM
# ==========================================
elif "Webcam" in menu:
    st.header("🎥 4. Webcam")

    # 🔧 เลือกโมเดลเหมือน Predict / Video
    model_path = st.text_input(
        "📂 Path Model",
        "runs/detect/my_custom_model/weights/best.pt"
    )
    conf_threshold = st.slider(
        "🎚️ Confidence",
        0.0, 1.0, 0.25, 0.05
    )

    run_cam = st.toggle("เปิดกล้อง")
    placeholder = st.empty()

    if run_cam:
        try:
            model = YOLO(model_path)  # ✅ ใช้ best.pt
        except:
            st.error("❌ ไม่พบไฟล์โมเดล")
            st.stop()

        cap = cv2.VideoCapture(0)

        while run_cam:
            ret, frame = cap.read()
            if not ret:
                break

            # 🔍 Predict
            results = model(frame, conf=conf_threshold)
            frame_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

            placeholder.image(frame_rgb, use_column_width=True)

        cap.release()


# ==========================================
# 5. VIDEO SCAN (ใช้ best.pt เหมือน Predict)
# ==========================================
elif "Video Scan" in menu:
    st.header("📹 5. Video Scan (Upload / YouTube)")

    import uuid

    # ---------- SESSION STATE ----------
    if "current_video_path" not in st.session_state:
        st.session_state.current_video_path = None
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # ---------- CONFIG ----------
    col1, col2 = st.columns(2)

    with col1:
        model_path = st.text_input(
            "📂 Path Model",
            "runs/detect/my_custom_model/weights/best.pt",
            key="vid_model_path"
        )
        conf_threshold = st.slider(
            "🎚️ Confidence",
            0.0, 1.0, 0.25, 0.05,
            key="vid_conf"
        )

    with col2:
        source_type = st.radio(
            "แหล่งที่มา",
            ["📁 Upload File", "🔴 YouTube URL"]
        )

        # ---------- UPLOAD FILE ----------
        if source_type == "📁 Upload File":
            video_file = st.file_uploader(
                "🎬 อัปโหลดไฟล์วิดีโอ",
                type=["mp4", "mov", "avi"]
            )
            if video_file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(video_file.read())
                st.session_state.current_video_path = tfile.name
                st.success("อัปโหลดสำเร็จ")

        # ---------- YOUTUBE ----------
        else:
            youtube_url = st.text_input("🔗 YouTube URL (รองรับ Shorts)")
            if st.button("📥 โหลดวิดีโอ"):
                if youtube_url:
                    with st.spinner("กำลังดาวน์โหลดวิดีโอ..."):
                        try:
                            # ลบไฟล์เก่าถ้ามี
                            if (
                                st.session_state.current_video_path
                                and os.path.exists(st.session_state.current_video_path)
                            ):
                                try:
                                    os.remove(st.session_state.current_video_path)
                                except:
                                    pass

                            unique_name = f"yt_{uuid.uuid4().hex}.mp4"
                            ydl_opts = {
                                "format": "best[ext=mp4]/best",
                                "outtmpl": os.path.join(BASE_DIR, unique_name),
                                "quiet": True,
                                "noplaylist": True,
                            }

                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(youtube_url, download=True)
                                filename = ydl.prepare_filename(info)
                                st.session_state.current_video_path = filename
                                st.success(f"โหลดเสร็จ: {info.get('title', 'Video')}")

                        except Exception as e:
                            st.error(f"โหลดไม่สำเร็จ: {e}")

    # ---------- VIDEO INFO ----------
    if st.session_state.current_video_path:
        st.info(
            f"📂 ไฟล์ปัจจุบัน: "
            f"{os.path.basename(st.session_state.current_video_path)}"
        )

        if st.button("▶️ เริ่ม / หยุด Scan", type="primary"):
            st.session_state.processing = not st.session_state.processing

        # ---------- PROCESSING ----------
        if st.session_state.processing:
            try:
                # โหลดโมเดล (best.pt)
                model = YOLO(model_path)

                cap = cv2.VideoCapture(st.session_state.current_video_path)
                st_frame = st.empty()

                while cap.isOpened() and st.session_state.processing:
                    ret, frame = cap.read()
                    if not ret:
                        st.session_state.processing = False
                        break

                    # 🔧 Resize แบบคงสัดส่วน (เหมาะกับ Shorts)
                    target_width = 600
                    h, w = frame.shape[:2]
                    scale = target_width / w
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (target_width, new_h))

                    # 🔍 Predict
                    results = model(frame, conf=conf_threshold)
                    plotted = results[0].plot()

                    frame_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                    st_frame.image(
                        frame_rgb,
                        caption="Scanning...",
                        width=target_width
                    )

                cap.release()

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {e}")
                st.session_state.processing = False

    else:
        st.warning("👈 กรุณาอัปโหลดไฟล์หรือใส่ลิงก์ YouTube ก่อน")
