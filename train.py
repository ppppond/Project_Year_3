import warnings
warnings.filterwarnings("ignore") 

def train_page():
    import streamlit as st
    import os
    import torch
    from ultralytics import YOLO
    from helpers.dataset_helper import create_yaml
    from config import DATASET_DIR
    
    # ==========================================
    # HEADER
    # ==========================================
    st.header("🚀 2. Train Model")

    # ==========================================
    # CLASS LIST
    # ==========================================
    st.subheader("📌 Class List")

    class_text = st.text_area(
        "ใส่ชื่อคลาส (1 บรรทัด ต่อ 1 class)",
        "\n".join(st.session_state.get("class_list", [])),
        height=120,
        placeholder="person\ncar\nmotorcycle"
    )

    classes = [c.strip() for c in class_text.split("\n") if c.strip()]

    if classes:
        st.info(f"พบ {len(classes)} classes : {classes}")
    else:
        st.warning("ยังไม่ได้ใส่ class")

    # ==========================================
    # CREATE YAML
    # ==========================================
    if st.button("📄 สร้าง data.yaml"):
        if not classes:
            st.error("❌ กรุณาใส่ Class อย่างน้อย 1 class")
        else:
            create_yaml(DATASET_DIR, classes)
            st.session_state["class_list"] = classes
            st.success("✅ สร้าง data.yaml สำเร็จ")

    st.divider()

    # ==========================================
    # TRAIN SETTINGS
    # ==========================================
    st.subheader("⚙️ Train Settings")

    epochs = st.number_input(
        "Epochs",
        min_value=10,
        max_value=300,
        value=30,
        step=10
    )

    batch = st.selectbox(
        "Batch Size",
        [4, 8, 16, 32],
        index=1
    )

    imgsz = st.selectbox(
        "Image Size",
        [416, 512, 640, 768],
        index=2
    )

    model_type = st.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0
    )

    st.divider()

    # ==========================================
    # START TRAIN
    # ==========================================
    if st.button("🚀 Start Training"):

        yaml_path = os.path.join(DATASET_DIR, "data.yaml")

        if not os.path.exists(yaml_path):
            st.error("❌ ไม่พบ data.yaml กรุณาสร้างก่อน")
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

        # --------------------------------------
        # Train Model
        # --------------------------------------
        with st.spinner("⏳ กำลัง Train Model..."):

            model = YOLO(model_type)

            model.train(
                data=yaml_path,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,  
                workers=2, 
                project="runs/detect",
                name="my_custom_model",
                exist_ok=True,
            )

        st.success("🎉 Train เสร็จเรียบร้อย!")
