import streamlit as st
import os
import math
from config import IMG_DIR, LABEL_DIR
from PIL import Image, ImageDraw


def preview_page():
    st.header("🖼️ Preview Labeled Images (With Bounding Boxes)")

    if not os.path.exists(IMG_DIR):
        st.warning("❌ IMG_DIR ไม่พบ")
        return

    labeled_imgs = sorted([
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not labeled_imgs:
        st.info("ยังไม่มีรูปที่บันทึก")
        return

    # ==============================
    # 🎯 เลือก Class
    # ==============================
    if "class_list" in st.session_state and st.session_state.class_list:
        selected_filter = st.selectbox(
            "🎯 เลือก Class ที่ต้องการแสดง",
            ["ทั้งหมด"] + st.session_state.class_list
        )
    else:
        st.warning("ไม่พบ class_list ใน session_state")
        return

    # ==============================
    # 🔎 Filter รูปจาก label ก่อน
    # ==============================
    filtered_imgs = []

    if selected_filter == "ทั้งหมด":
        filtered_imgs = labeled_imgs
    else:
        selected_id = st.session_state.class_list.index(selected_filter)

        for img_file in labeled_imgs:
            label_file = img_file.rsplit(".", 1)[0] + ".txt"
            label_path = os.path.join(LABEL_DIR, label_file)

            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(float(parts[0]))
                    if cid == selected_id:
                        filtered_imgs.append(img_file)
                        break  # เจอแล้วไม่ต้องเช็คต่อ

    if not filtered_imgs:
        st.warning("ไม่พบรูปที่มี class นี้")
        return

    # ==============================
    # 📄 Pagination
    # ==============================
    images_per_page = 50
    total_images = len(filtered_imgs)
    total_pages = math.ceil(total_images / images_per_page)

    page = st.number_input(
        "เลือกหน้า",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )

    start_idx = (page - 1) * images_per_page
    end_idx = start_idx + images_per_page
    page_images = filtered_imgs[start_idx:end_idx]

    st.markdown(
        f"แสดงรูป {start_idx + 1} - {min(end_idx, total_images)} จาก {total_images} รูป"
    )

    # ==============================
    # 🖼️ แสดงรูป (5 รูปต่อแถว)
    # ==============================
    cols = st.columns(5)

    for i, img_file in enumerate(page_images):
        with cols[i % 5]:

            img_path = os.path.join(IMG_DIR, img_file)
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)

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
                        class_id = int(cid)

                        left = (xc - nw / 2) * w
                        top = (yc - nh / 2) * h
                        right = (xc + nw / 2) * w
                        bottom = (yc + nh / 2) * h

                        draw.rectangle(
                            [left, top, right, bottom],
                            outline="red",
                            width=2
                        )

                        class_name = (
                            st.session_state.class_list[class_id]
                            if class_id < len(st.session_state.class_list)
                            else f"Class {class_id}"
                        )

                        draw.text(
                            (left, max(0, top - 12)),
                            class_name,
                            fill="red"
                        )

            st.image(img, caption=img_file, use_column_width=True)
