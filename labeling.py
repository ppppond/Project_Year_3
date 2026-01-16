def labeling_page():
    import streamlit as st
    import pandas as pd
    import os
    from PIL import Image
    from streamlit_drawable_canvas import st_canvas
    from config import IMG_DIR, LABEL_DIR
    from helpers.image_helper import transform_image

# ==========================================
# 1. LABELING (MULTI IMAGE)
# ==========================================
    st.header("🖌️ 1. Labeling: Multi-Image Annotation")

    col1, col2 = st.columns([1, 3])

    # ---------- LEFT ----------
    with col1:
        st.markdown("### ⚙️ ตั้งค่า Class")

        if "class_list" not in st.session_state:
            st.session_state.class_list = ["dog", "cat", "person"]

        class_text = st.text_area(
            "รายชื่อวัตถุ (บรรทัดละชื่อ)",
            "\n".join(st.session_state.class_list),
            height=120,
        )
        st.session_state.class_list = [
            c.strip() for c in class_text.split("\n") if c.strip()
        ]

        selected_class = st.selectbox(
            "👉 กำลังจะวาด:",
            st.session_state.class_list,
        )

        stroke_width = st.slider("ความหนาเส้น", 1, 5, 2)

        st.divider()
        st.markdown("### 🖼️ Image Config")

        rotate_angle = st.slider("🔄 Rotate", -180, 180, 0, step=5)
        flip_h = st.checkbox("↔️ Flip Horizontal")
        flip_v = st.checkbox("↕️ Flip Vertical")
        brightness = st.slider("☀️ Brightness", 0.2, 2.0, 1.0)
        contrast = st.slider("🎚️ Contrast", 0.2, 2.0, 1.0)
        color = st.slider("🎨 Color", 0.2, 2.0, 1.0)

    # ---------- RIGHT ----------
    with col2:
        uploaded_files = st.file_uploader(
            "📂 อัปโหลดรูปภาพ (หลายรูปได้)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            file_names = [f.name for f in uploaded_files]

            selected_idx = st.selectbox(
                "🖼️ เลือกรูปที่จะ Label",
                range(len(file_names)),
                format_func=lambda i: file_names[i],
            )

            uploaded_file = uploaded_files[selected_idx]

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

            w, h = image.size

            canvas_key = f"canvas_{uploaded_file.name}_{rotate_angle}_{flip_h}_{flip_v}_{brightness}_{contrast}_{color}"

            canvas = st_canvas(
                fill_color="rgba(0, 150, 255, 0.2)",
                stroke_width=stroke_width,
                stroke_color="#0066CC",
                background_image=image,
                update_streamlit=True,
                height=500,
                drawing_mode="rect",
                key=canvas_key,
            )

            if canvas.json_data is not None:
                objects = pd.json_normalize(canvas.json_data["objects"])
                if not objects.empty:
                    st.success(f"✅ พบ {len(objects)} กรอบ")

                    if st.button("💾 บันทึกข้อมูล (Save)", type="primary"):
                        suffix = []
                        if rotate_angle != 0:
                            suffix.append(f"rot{rotate_angle}")
                        if flip_h:
                            suffix.append("fh")
                        if flip_v:
                            suffix.append("fv")
                        if brightness != 1.0:
                            suffix.append(f"b{brightness}")
                        if contrast != 1.0:
                            suffix.append(f"c{contrast}")
                        if color != 1.0:
                            suffix.append(f"col{color}")
                        suffix = "_".join(suffix) if suffix else "orig"

                        img_name = f"{os.path.splitext(uploaded_file.name)[0]}_{suffix}.jpg"
                        label_name = img_name.replace(".jpg", ".txt")

                        image.save(os.path.join(IMG_DIR, img_name))

                        lines = []
                        for _, row in objects.iterrows():
                            xc = (row["left"] + row["width"] / 2) / w
                            yc = (row["top"] + row["height"] / 2) / h
                            nw = row["width"] / w
                            nh = row["height"] / h
                            cid = st.session_state.class_list.index(selected_class)
                            lines.append(
                                f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
                            )

                        with open(os.path.join(LABEL_DIR, label_name), "w") as f:
                            f.write("\n".join(lines))

                        st.toast("บันทึกเรียบร้อย 🎉")
