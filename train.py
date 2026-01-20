def train_page():
    import streamlit as st
    import os
    from ultralytics import YOLO
    from helpers.dataset_helper import create_yaml
    from config import DATASET_DIR

    # ==========================================
    # 2. TRAIN
    # ==========================================

    st.header("🚀 2. Train Model")

    # -----------------------------
    # CLASS LIST
    # -----------------------------
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

    # -----------------------------
    # CREATE data.yaml
    # -----------------------------
    if st.button("📄 สร้าง data.yaml"):
        if not classes:
            st.error("❌ กรุณาใส่ Class อย่างน้อย 1 class")
        else:
            create_yaml(DATASET_DIR, classes)
            st.session_state["class_list"] = classes
            st.success("✅ สร้าง data.yaml สำเร็จ")

    st.divider()

    # -----------------------------
    # TRAIN SETTING
    # -----------------------------
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

    # -----------------------------
    # START TRAIN
    # -----------------------------
    if st.button("🚀 Start Training"):
        yaml_path = os.path.join(DATASET_DIR, "data.yaml")

        if not os.path.exists(yaml_path):
            st.error("❌ ไม่พบ data.yaml กรุณาสร้างก่อน")
            return

        with st.spinner("⏳ กำลัง Train Model..."):
            model = YOLO(model_type)
            model.train(
                data=yaml_path,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                project="runs/detect",
                name="my_custom_model",
                exist_ok=True,
            )

        st.success("🎉 Train เสร็จเรียบร้อย!")
