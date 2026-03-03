def predict_page():
    import streamlit as st
    import cv2
    from PIL import Image
    from ultralytics import YOLO

# ==========================================
# 3. PREDICT (แก้ไข: สี + Confidence Slider)
# ==========================================
    st.header("🔮 Predict")

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
