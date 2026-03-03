def video_scan_page():
    import streamlit as st
    import cv2
    import os
    import tempfile
    import yt_dlp
    import uuid
    from ultralytics import YOLO
    from config import BASE_DIR

    # ==========================================
    # 5. VIDEO SCAN (ใช้ best.pt เหมือน Predict)
    # ==========================================

    st.header("📹 Video Scan (Upload / YouTube)")

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

                            # 🔥 FIX SABR STREAMING
                            ydl_opts = {
                                "format": "bv*[ext=mp4]+ba[ext=m4a]/b",
                                "outtmpl": os.path.join(BASE_DIR, unique_name),
                                "noplaylist": True,
                                "quiet": True,
                                "merge_output_format": "mp4",
                                "extractor_args": {
                                    "youtube": {
                                        "player_client": ["android"]
                                    }
                                }
                            }

                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(youtube_url, download=True)
                                filename = ydl.prepare_filename(info)

                                # ✅ ป้องกันไฟล์ว่าง
                                if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                                    raise Exception("ไฟล์วิดีโอว่าง (0 bytes)")

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

        # ---------- EXPAND VIDEO BUTTON ----------
        expand_video = st.checkbox("🔍 ขยายวิดีโอใหญ่", value=False)
        target_width = 1200 if expand_video else 600

        # ---------- PROCESSING ----------
        if st.session_state.processing:
            try:
                # โหลดโมเดล (best.pt)
                model = YOLO(model_path)

                cap = cv2.VideoCapture(st.session_state.current_video_path)
                st_frame = st.empty()

                if not cap.isOpened():
                    raise Exception("ไม่สามารถเปิดไฟล์วิดีโอได้")

                while cap.isOpened() and st.session_state.processing:
                    ret, frame = cap.read()
                    if not ret:
                        st.session_state.processing = False
                        break

                    # 🔧 Resize คงสัดส่วน
                    h, w = frame.shape[:2]
                    scale = target_width / w
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (target_width, new_h))

                    # 🔍 YOLO Predict
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
