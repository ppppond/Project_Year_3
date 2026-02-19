def labeling_pagev2():
    import streamlit as st
    import pandas as pd
    import os
    import hashlib
    import random
    from PIL import Image, ImageDraw
    from streamlit_drawable_canvas import st_canvas
    from config import IMG_DIR, LABEL_DIR
    from helpers.image_helper import transform_image

    CLASSES_FILE = "classes.txt"
    MAX_WIDTH = 1000

    # ==============================
    # Load / Save Classes
    # ==============================
    def load_classes():
        if not os.path.exists(CLASSES_FILE):
            return []
        with open(CLASSES_FILE, "r", encoding="utf-8") as f:
            return [c.strip() for c in f.readlines() if c.strip()]

    def save_classes(class_list):
        with open(CLASSES_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(class_list))

    # ==============================
    # Draw Preview Boxes
    # ==============================
    def draw_boxes_preview(image, objects, class_per_box, class_list, scale):
        preview = image.copy()
        draw = ImageDraw.Draw(preview)

        random.seed(42)
        class_colors = {
            cls: tuple(random.randint(60, 255) for _ in range(3))
            for cls in class_list
        }

        for idx, row in objects.iterrows():
            left = row["left"] / scale
            top = row["top"] / scale
            width = row["width"] / scale
            height = row["height"] / scale

            x1 = left
            y1 = top
            x2 = left + width
            y2 = top + height

            cls_name = class_per_box[idx]
            color = class_colors[cls_name]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            bbox = draw.textbbox((0, 0), cls_name)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            draw.rectangle(
                [x1, y1 - text_h - 6, x1 + text_w + 6, y1],
                fill=color
            )
            draw.text((x1 + 3, y1 - text_h - 3), cls_name, fill="black")

        return preview, class_colors

    # ==============================
    # UI START
    # ==============================
    st.header("🖌️ 1. Labeling: Multi-Image Annotation")

    col1, col2 = st.columns([1, 3])

    # =========================================================
    # LEFT PANEL
    # =========================================================
    with col1:

        st.markdown("### ⚙️ ตั้งค่า Class")

        if "class_list" not in st.session_state:
            st.session_state.class_list = load_classes()

        class_text = st.text_area(
            "รายชื่อวัตถุ (บรรทัดละชื่อ)",
            "\n".join(st.session_state.class_list),
            height=120,
        )

        new_class_list = [c.strip() for c in class_text.split("\n") if c.strip()]

        if st.button("💾 Save Classes"):
            if not new_class_list:
                st.warning("ต้องมีอย่างน้อย 1 class")
            else:
                save_classes(new_class_list)
                st.session_state.class_list = new_class_list
                st.success("บันทึก classes.txt แล้ว")
                st.rerun()

        if not new_class_list:
            st.warning("⚠ กรุณาเพิ่มอย่างน้อย 1 class")
            return

        stroke_width = st.slider("ความหนาเส้น", 1, 6, 2)

        st.divider()
        st.markdown("### 🖼️ Image Config")

        defaults = {
            "rotate_angle": 0,
            "flip_h": False,
            "flip_v": False,
            "brightness": 1.0,
            "contrast": 1.0,
            "color": 1.0,
        }

        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        rotate_angle = st.slider("🔄 Rotate", -180, 180, step=5, key="rotate_angle")
        flip_h = st.checkbox("↔️ Flip Horizontal", key="flip_h")
        flip_v = st.checkbox("↕️ Flip Vertical", key="flip_v")
        brightness = st.slider("☀️ Brightness", 0.2, 2.0, key="brightness")
        contrast = st.slider("🎚️ Contrast", 0.2, 2.0, key="contrast")
        color = st.slider("🎨 Color", 0.2, 2.0, key="color")

        def reset_image_config():
            for k, v in defaults.items():
                st.session_state[k] = v

        st.button("♻️ Reset Image Config", on_click=reset_image_config)

    # =========================================================
    # RIGHT PANEL
    # =========================================================
    with col2:

        uploaded_files = st.file_uploader(
            "📂 อัปโหลดรูปภาพ (หลายรูปได้)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            return

        file_names = [f.name for f in uploaded_files]

        selected_idx = st.selectbox(
            "🖼️ เลือกรูปที่จะ Label",
            range(len(file_names)),
            format_func=lambda i: file_names[i],
        )

        uploaded_file = uploaded_files[selected_idx]
        uploaded_file.seek(0)
        file_bytes = uploaded_file.getvalue()

        orig_image = Image.open(uploaded_file).convert("RGB")

        image = transform_image(
            orig_image,
            rotate_angle,
            flip_h,
            flip_v,
            brightness,
            contrast,
            color,
        )

        orig_w, orig_h = image.size

        if orig_w > MAX_WIDTH:
            scale = MAX_WIDTH / orig_w
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            display_image = image.resize((new_w, new_h))
        else:
            scale = 1.0
            new_w, new_h = orig_w, orig_h
            display_image = image

        canvas_hash = hashlib.md5(file_bytes).hexdigest()
        canvas_key = f"canvas_{canvas_hash}"

        canvas = st_canvas(
            fill_color="rgba(0,150,255,0.2)",
            stroke_width=stroke_width,
            stroke_color="#0066CC",
            background_image=display_image,
            update_streamlit=True,
            height=new_h,
            width=new_w,
            drawing_mode="rect",
            key=canvas_key,
        )

        # ==========================
        # Multi-Class per Box
        # ==========================
        if canvas.json_data is not None:

            objects = pd.json_normalize(canvas.json_data["objects"])

            if not objects.empty:

                st.success(f"✅ พบ {len(objects)} กรอบ")

                st.markdown("### 🏷️ กำหนด Class ให้แต่ละกรอบ")

                class_per_box = []

                for i in range(len(objects)):
                    selected = st.selectbox(
                        f"กรอบที่ {i+1}",
                        new_class_list,
                        key=f"class_box_{i}"
                    )
                    class_per_box.append(selected)

                preview_image, class_colors = draw_boxes_preview(
                    image,
                    objects,
                    class_per_box,
                    new_class_list,
                    scale
                )

                st.markdown("### 🔍 Preview")
                st.image(preview_image, use_column_width=True)

                st.markdown("### 🎨 Class Legend")
                for cls, color in class_colors.items():
                    st.markdown(
                        f"<span style='color: rgb{color}; font-weight: bold;'>■</span> {cls}",
                        unsafe_allow_html=True
                    )

                if st.button("💾 บันทึกข้อมูล (Save)", type="primary"):

                    img_name = uploaded_file.name
                    label_name = os.path.splitext(img_name)[0] + ".txt"

                    os.makedirs(IMG_DIR, exist_ok=True)
                    os.makedirs(LABEL_DIR, exist_ok=True)

                    image.save(os.path.join(IMG_DIR, img_name))

                    lines = []

                    for idx, row in objects.iterrows():
                        left = row["left"] / scale
                        top = row["top"] / scale
                        width = row["width"] / scale
                        height = row["height"] / scale

                        xc = (left + width / 2) / orig_w
                        yc = (top + height / 2) / orig_h
                        nw = width / orig_w
                        nh = height / orig_h

                        cid = new_class_list.index(class_per_box[idx])

                        lines.append(
                            f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
                        )

                    with open(os.path.join(LABEL_DIR, label_name), "w") as f:
                        f.write("\n".join(lines))

                    st.toast("บันทึกเรียบร้อย 🎉")
