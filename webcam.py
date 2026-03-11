def webcam_page():
    import streamlit as st
    import cv2
    from ultralytics import YOLO
    import time

    st.header("🎥 Webcam (ไม่ใช้ Thread)")

    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False

    col1, col2 = st.columns(2)
    with col1:
        model_path = st.text_input("📂 Path Model", "runs/detect/my_custom_model/weights/best.pt")
    with col2:
        conf_threshold = st.slider("🎚️ Confidence", 0.0, 1.0, 0.25, 0.05)

    mirror_view = st.checkbox("🪞 Mirror View ", value=True)
    camera_index = st.number_input("📷 Camera Index", min_value=0, max_value=5, value=0)

    placeholder = st.empty()

    # ปุ่มเปิด/ปิด
    if st.button("▶️ เปิด / ปิด กล้อง"):
        st.session_state.cam_running = not st.session_state.cam_running

    # Run webcam ใน main thread
    if st.session_state.cam_running:
        try:
            model = YOLO(model_path)
        except:
            st.error("❌ ไม่พบไฟล์โมเดล")
            st.session_state.cam_running = False
            st.stop()

        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.error("❌ เปิดกล้องไม่สำเร็จ")
            st.session_state.cam_running = False
            st.stop()

        while st.session_state.cam_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            results = model(frame, conf=conf_threshold)
            plotted = results[0].plot()
            if mirror_view:
                plotted = cv2.flip(plotted, 1)

            frame_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, use_column_width=True)

            # รอ 0.03 วินาที เพื่อไม่ให้ CPU พุ่ง
            time.sleep(0.03)

        cap.release()
        placeholder.empty()
