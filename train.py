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

