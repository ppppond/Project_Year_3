def predict_page():
    import streamlit as st
    import cv2
    import os
    from PIL import Image
    from ultralytics import YOLO
    import tkinter as tk
    from tkinter import filedialog

    # ==========================================
    # 3. PREDICT (ปรับปรุงระบบเลือกไฟล์แบบ Folder/File)
    # ==========================================
    st.header("🔮 Predict")

    col_cfg, col_files = st.columns([1, 2])
    
    with col_cfg:
        st.markdown("### ⚙️ Model Config")
        model_path = st.text_input(
            "📂 Path Model",
            "runs/detect/my_custom_model/weights/best.pt"
        )
        conf_threshold = st.slider("🎚️ ความมั่นใจ (Confidence)", 0.0, 1.0, 0.25, 0.05)

    with col_files:
        st.markdown("### 📁 Select Images")
        
        # --- ระบบเลือก Folder และ File (เหมือนหน้า Labeling) ---
        if "pred_selected_files" not in st.session_state:
            st.session_state.pred_selected_files = []
        if "pred_target_folder" not in st.session_state:
            st.session_state.pred_target_folder = ""

        # ปรับปุ่มให้ชิดกันทางซ้าย
        col_btn1, col_btn2, col_spacer = st.columns([0.25, 0.25, 0.5])
        
        with col_btn1:
            if st.button("📁 Open Folder...", use_container_width=True):
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                folder_selected = filedialog.askdirectory(master=root)
                root.destroy()
                if folder_selected:
                    st.session_state.pred_target_folder = folder_selected
                    st.session_state.pred_selected_files = sorted([
                        os.path.join(folder_selected, f) for f in os.listdir(folder_selected)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ])
                    st.rerun()

        with col_btn2:
            if st.button("🖼️ Open File...", use_container_width=True):
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                files_selected = filedialog.askopenfilenames(
                    master=root,
                    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
                )
                root.destroy()
                if files_selected:
                    st.session_state.pred_selected_files = list(files_selected)
                    st.session_state.pred_target_folder = os.path.dirname(files_selected[0])
                    st.rerun()

        if not st.session_state.pred_selected_files:
            st.info("💡 เลือก 'Open Folder' หรือ 'Open File' เพื่อเริ่มการทำนาย")
            return

        # --- แสดงรายการไฟล์ให้เลือก ---
        file_paths = st.session_state.pred_selected_files
        file_names_only = [os.path.basename(p) for p in file_paths]

        selected_idx = st.selectbox(
            f"📄 รายการไฟล์ (พบ {len(file_names_only)} รูป)",
            range(len(file_names_only)),
            format_func=lambda i: file_names_only[i],
        )

        img_path = file_paths[selected_idx]

        if st.button("🔍 เริ่มทำนาย (Predict)", type="primary", use_container_width=True):
            try:
                model = YOLO(model_path)
            except Exception:
                st.error(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
                st.stop()

            # โหลดรูปและรันโมเดล
            pil_img = Image.open(img_path).convert("RGB")
            results = model(pil_img, conf=conf_threshold)

            # ประมวลผลภาพผลลัพธ์
            res_plotted_bgr = results[0].plot()
            res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)

            # แสดงผล
            st.divider()
            st.image(res_plotted_rgb, caption=f"ผลลัพธ์: {file_names_only[selected_idx]}", use_container_width=True)
            
            # สรุปจำนวน
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"✅ เจอวัตถุทั้งหมด {len(boxes)} ชิ้น")
            else:
                st.warning("⚠️ ไม่เจอวัตถุ (ลองลดค่า Confidence ลงดูนะครับ)")