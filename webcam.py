def webcam_page():
    import streamlit as st
    import cv2
    from ultralytics import YOLO

# ==========================================
# 4. WEBCAM
# ==========================================

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


