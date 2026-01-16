import streamlit as st
import os
from config import IMG_DIR, LABEL_DIR
from PIL import Image, ImageDraw

def preview_page():
    st.header("🖼️ Preview Labeled Images (With Bounding Boxes)")

    if not os.path.exists(IMG_DIR):
        st.warning("❌ IMG_DIR ไม่พบ")
        return

    labeled_imgs = [
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not labeled_imgs:
        st.info("ยังไม่มีรูปที่บันทึก")
        return

    st.markdown("### รูปที่ Label เสร็จแล้ว")
    cols = st.columns(3)

    for i, img_file in enumerate(labeled_imgs):
        with cols[i % 3]:
            img_path = os.path.join(IMG_DIR, img_file)
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            # โหลด label
            label_file = img_file.rsplit(".", 1)[0] + ".txt"
            label_path = os.path.join(LABEL_DIR, label_file)
            if os.path.exists(label_path):
                w, h = img.size
                with open(label_path, "r") as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cid, xc, yc, nw, nh = map(float, parts)
                            # แปลง normalized coordinates เป็น pixel
                            left = (xc - nw/2) * w
                            top = (yc - nh/2) * h
                            right = (xc + nw/2) * w
                            bottom = (yc + nh/2) * h
                            draw.rectangle([left, top, right, bottom], outline="red", width=2)
                            draw.text((left, top-10), f"Class {int(cid)}", fill="red")

            st.image(img, caption=img_file, use_column_width=True)
