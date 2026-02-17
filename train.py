def train_page():
    import streamlit as st
    import os
    import torch
    from ultralytics import YOLO
    from helpers.dataset_helper import create_yaml
    # ตรวจสอบว่าไฟล์ config มีอยู่จริงและ import ได้
    # ถ้าไม่มีให้แก้เป็น path string ตรงๆ หรือสร้างไฟล์ config.py
    try:
        from config import DATASET_DIR
    except ImportError:
        DATASET_DIR = "datasets" # ค่า Default กรณี import ไม่ได้

    # ==========================================
    # HEADER
    # ==========================================
    st.header("🚀 2. Train Model")

    # ==========================================
    # CLASS LIST
    # ==========================================
    st.subheader("📌 Class List")

    # ดึงค่าเดิมมาแสดงถ้ามี
    default_classes = "\n".join(st.session_state.get("class_list", []))
    
    class_text = st.text_area(
        "ใส่ชื่อคลาส (1 บรรทัด ต่อ 1 class)",
        value=default_classes,
        height=120,
        placeholder="person\ncar\nmotorcycle"
    )

    # แปลง text เป็น list โดยตัดช่องว่างและบรรทัดว่างออก
    classes = [c.strip() for c in class_text.split("\n") if c.strip()]

    if classes:
        st.info(f"พบ {len(classes)} classes : {classes}")
        # อัปเดต session state ทันทีที่มีการเปลี่ยนแปลง (Optional)
        st.session_state["class_list"] = classes
    else:
        st.warning("ยังไม่ได้ใส่ class")

    # ==========================================
    # CREATE YAML
    # ==========================================
    if st.button("📄 สร้าง data.yaml"):
        if not classes:
            st.error("❌ กรุณาใส่ Class อย่างน้อย 1 class")
        else:
            # สร้างโฟลเดอร์ datasets ถ้ายังไม่มี
            os.makedirs(DATASET_DIR, exist_ok=True)
            
            create_yaml(DATASET_DIR, classes)
            st.success(f"✅ สร้าง data.yaml สำเร็จที่ {DATASET_DIR}")

    st.divider()

    # ==========================================
    # TRAIN SETTINGS
    # ==========================================
    st.subheader("⚙️ Train Settings")

    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=1000,
            value=30,
            step=10
        )
        
        imgsz = st.selectbox(
            "Image Size",
            [416, 512, 640, 768, 1024],
            index=2
        )

    with col2:
        batch = st.selectbox(
            "Batch Size",
            [4, 8, 16, 32, 64],
            index=2  # แนะนำ 16 สำหรับ M4 Air (ถ้า RAM 16GB)
        )

        model_type = st.selectbox(
            "YOLO Model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
            index=0
        )

    st.divider()

    # ==========================================
    # START TRAIN
    # ==========================================
    if st.button("🚀 Start Training", type="primary"):

        yaml_path = os.path.join(DATASET_DIR, "data.yaml")

        if not os.path.exists(yaml_path):
            st.error(f"❌ ไม่พบไฟล์ {yaml_path} กรุณากดสร้าง data.yaml ก่อน")
            return

        # --------------------------------------
        # ตรวจสอบ GPU (รองรับ NVIDIA / AMD (ROCm) / Mac M-Series)
        # --------------------------------------
        device = "cpu"  # Default

        # CUDA (NVIDIA) หรือ ROCm (AMD) สำหรับ Linux ที่ติดตั้ง PyTorch ที่รองรับ ROCm
        if torch.cuda.is_available():
            # ถ้าเป็น ROCm build จะมี torch.version.hip
            if getattr(torch.version, "hip", None) is not None:
                # ROCm (AMD) - ใช้ same device indexing as CUDA (0, 1, ...)
                device = 0
                st.success("🚀 ใช้ AMD GPU (ROCm)")
            else:
                device = 0
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = "CUDA GPU"
                st.success(f"🚀 ใช้ NVIDIA GPU: {gpu_name}")

        elif torch.backends.mps.is_available():
            # นี่คือส่วนสำคัญสำหรับ Mac M4 / M-Series
            device = "mps"
            st.success("🚀 ใช้ Apple Metal Performance Shaders (MPS / M-Series GPU) 🍏")

        else:
            # สำหรับ Windows AMD (DirectML) หรือกรณีอื่นๆ ให้ตรวจสอบเพิ่มเติมด้วยตัวเอง
            st.warning("⚠️ ไม่พบ GPU ที่รองรับ (CUDA/MPS/ROCm). ระบบจะใช้ CPU แทน (อาจจะช้าหน่อย)")

        # --------------------------------------
        # Train Model
        # --------------------------------------
        status_text = st.empty()
        status_text.text("⏳ กำลังเตรียมการ Train...")
        
        progress_bar = st.progress(0)

        try:
            with st.spinner(f"กำลัง Train {epochs} epochs บน {device}..."):

                # โหลดโมเดล
                model = YOLO(model_type)

                # เริ่ม Train
                results = model.train(
                    data=yaml_path,
                    epochs=epochs,
                    batch=batch,
                    imgsz=imgsz,
                    device=device,  # ส่งค่า device ที่เช็คมา (mps/0/cpu)
                    project="runs/detect",
                    name="train_result",
                    exist_ok=True, # ทับโฟลเดอร์เดิม (ถ้าอยากให้สร้างใหม่เรื่อยๆ ให้เอาออก)
                    plots=True
                )
            
            progress_bar.progress(100)
            status_text.text("✅ Train เสร็จสิ้น!")
            st.success("🎉 Train Model เสร็จเรียบร้อย!")
            
            # แสดงผลลัพธ์เบื้องต้น (ถ้ามีไฟล์)
            results_dir = os.path.join("runs/detect", "train_result")
            confusion_matrix = os.path.join(results_dir, "confusion_matrix.png")
            results_png = os.path.join(results_dir, "results.png")

            if os.path.exists(results_png):
                st.image(results_png, caption="Training Results")
            
            st.info(f"📁 ผลลัพธ์ถูกบันทึกอยู่ที่: {results_dir}")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดขณะ Train: {e}")
            # print error ใน terminal ด้วยเพื่อ debug ง่ายขึ้น
            print(f"Error details: {e}")