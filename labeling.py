def labeling_page():
    import streamlit as st
    import pandas as pd
    import os
    import json
    from PIL import Image
    from streamlit_drawable_canvas import st_canvas
    from config import IMG_DIR, LABEL_DIR
    from helpers.image_helper import transform_image

    st.header("🖌️ Labeling")

    label_mode = st.radio(
        "📐 เลือกประเภท Label",
        ["🔲 Bounding Box", "📏 Line / Polyline", "✏️ Polygon"],
        horizontal=True,
    )

    st.divider()
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### ⚙️ ตั้งค่า Class")

        state_key = (
            "class_list"      if "Bounding" in label_mode else
            "line_class_list" if "Line"     in label_mode else
            "poly_class_list"
        )
        default_classes = {
            "class_list":      ["dog", "cat", "person"],
            "line_class_list": ["wire", "cable", "pipe", "crack", "weld_seam"],
            "poly_class_list": ["panel", "component", "region", "defect_area"],
        }
        if state_key not in st.session_state:
            st.session_state[state_key] = default_classes[state_key]

        class_text = st.text_area(
            "รายชื่อวัตถุ (บรรทัดละชื่อ)",
            "\n".join(st.session_state[state_key]),
            height=150,
            key=f"{state_key}_textarea",
        )
        st.session_state[state_key] = [c.strip() for c in class_text.split("\n") if c.strip()]

        selected_class = st.selectbox(
            "👉 กำลังจะวาด:",
            st.session_state[state_key],
            key=f"{state_key}_selectbox",
        )

        line_type = None

        if "Line" in label_mode:
            st.divider()
            st.markdown("### 📐 รูปแบบเส้น")
            line_type = st.radio(
                "ประเภทเส้น",
                ["📏 Line (2 จุด)", "〰️ Polyline (หลายจุด)"],
                key="line_type_radio",
            )
            stroke_color = st.color_picker("สีเส้น", "#FF4500", key="line_color_picker")
            drawing_mode = "line" if "Line (2" in line_type else "freedraw"
        elif "Polygon" in label_mode:
            st.divider()
            st.markdown("### ✏️ Polygon")
            st.info("คลิกวางจุด → ดับเบิลคลิกเพื่อปิด Polygon")
            stroke_color = st.color_picker("สีเส้น", "#00CC44", key="poly_color_picker")
            drawing_mode = "polygon"
        else:
            stroke_color = "#0066CC"
            drawing_mode = "rect"

        stroke_width = st.slider("ความหนาเส้น", 1, 8, 2, key="stroke_w")

        st.divider()
        st.markdown("### 🖼️ Image Config")
        rotate_angle = st.slider("🔄 Rotate", -180, 180, 0, step=5, key="rotate")
        flip_h = st.checkbox("↔️ Flip Horizontal", key="flip_h")
        flip_v = st.checkbox("↕️ Flip Vertical", key="flip_v")
        brightness = st.slider("☀️ Brightness", 0.2, 2.0, 1.0, key="brightness")
        contrast   = st.slider("🎚️ Contrast",   0.2, 2.0, 1.0, key="contrast")
        color      = st.slider("🎨 Color",       0.2, 2.0, 1.0, key="color")

    with col2:
        uploaded_files = st.file_uploader(
            "📂 อัปโหลดรูปภาพ (หลายรูปได้)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="main_uploader",
        )

        if uploaded_files:
            file_names = [f.name for f in uploaded_files]
            selected_idx = st.selectbox(
                "🖼️ เลือกรูปที่จะ Label",
                range(len(file_names)),
                format_func=lambda i: file_names[i],
                key="file_select",
            )
            uploaded_file = uploaded_files[selected_idx]

            orig_image = Image.open(uploaded_file).convert("RGB")
            image = transform_image(orig_image, rotate_angle, flip_h, flip_v, brightness, contrast, color)
            img_w, img_h = image.size

            CANVAS_H = 500
            CANVAS_W = int(CANVAS_H * img_w / img_h)

            canvas_key = (
                f"canvas_{label_mode}_{uploaded_file.name}_"
                f"{rotate_angle}_{flip_h}_{flip_v}_{brightness}_{contrast}_{color}"
            )

            canvas = st_canvas(
                fill_color="rgba(0, 150, 255, 0.2)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=image,
                update_streamlit=True,
                height=CANVAS_H,
                width=CANVAS_W,
                drawing_mode=drawing_mode,
                key=canvas_key,
            )

            if canvas.json_data is not None:
                objects = pd.json_normalize(canvas.json_data["objects"])

                # DEBUG
                with st.expander("🐛 Debug: raw canvas data"):
                    st.write("objects count:", len(canvas.json_data["objects"]))
                    st.json(canvas.json_data)

                # แสดงสถานะ
                if not objects.empty:
                    st.success(f"✅ พบ {len(objects)} {'กรอบ' if 'Bounding' in label_mode else 'รูปร่าง'}")
                else:
                    st.info("วาดแล้วกด Save ได้เลย")

                # ปุ่ม Save อยู่นอก if not empty → โชว์เสมอ
                if st.button("💾 บันทึกข้อมูล (Save)", type="primary", key="save_btn"):
                    if objects.empty:
                        st.warning("⚠️ ยังไม่มีข้อมูล กรุณาวาดก่อน")
                    else:
                        try:
                            suffix = []
                            if rotate_angle != 0: suffix.append(f"rot{rotate_angle}")
                            if flip_h:            suffix.append("fh")
                            if flip_v:            suffix.append("fv")
                            if brightness != 1.0: suffix.append(f"b{brightness}")
                            if contrast != 1.0:   suffix.append(f"c{contrast}")
                            if color != 1.0:      suffix.append(f"col{color}")
                            suffix = "_".join(suffix) if suffix else "orig"

                            img_name = f"{os.path.splitext(uploaded_file.name)[0]}_{suffix}.jpg"
                            image.save(os.path.join(IMG_DIR, img_name))
                            cid = st.session_state[state_key].index(selected_class)

                            # --- Bounding Box → YOLO .txt ---
                            if "Bounding" in label_mode:
                                label_name = img_name.replace(".jpg", ".txt")
                                lines = []
                                for _, row in objects.iterrows():
                                    sx   = float(row.get("scaleX", 1))
                                    sy   = float(row.get("scaleY", 1))
                                    left = float(row["left"])
                                    top  = float(row["top"])
                                    rw   = float(row["width"])  * sx
                                    rh   = float(row["height"]) * sy
                                    xc   = (left + rw / 2) / CANVAS_W
                                    yc   = (top  + rh / 2) / CANVAS_H
                                    nw_  = rw / CANVAS_W
                                    nh_  = rh / CANVAS_H
                                    lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw_:.6f} {nh_:.6f}")
                                with open(os.path.join(LABEL_DIR, label_name), "w") as f:
                                    f.write("\n".join(lines))

                            # --- Line / Polyline → JSON ---
                            elif "Line" in label_mode:
                                label_name = img_name.replace(".jpg", "_line.json")
                                annotations = []
                                for _, row in objects.iterrows():
                                    if line_type and "Line (2" in line_type:
                                        x1 = float(row.get("x1", row.get("left", 0))) / CANVAS_W
                                        y1 = float(row.get("y1", row.get("top",  0))) / CANVAS_H
                                        x2 = float(row.get("x2", row.get("left", 0) + row.get("width",  0))) / CANVAS_W
                                        y2 = float(row.get("y2", row.get("top",  0) + row.get("height", 0))) / CANVAS_H
                                        annotations.append({
                                            "class_id": cid, "class_name": selected_class,
                                            "type": "line",
                                            "points": [[round(x1,6), round(y1,6)], [round(x2,6), round(y2,6)]],
                                        })
                                    else:
                                        raw_path = row.get("path", [])
                                        points = [
                                            [round(float(cmd[1])/CANVAS_W, 6), round(float(cmd[2])/CANVAS_H, 6)]
                                            for cmd in raw_path if len(cmd) >= 3
                                        ]
                                        if points:
                                            annotations.append({
                                                "class_id": cid, "class_name": selected_class,
                                                "type": "polyline", "points": points,
                                            })
                                with open(os.path.join(LABEL_DIR, label_name), "w") as f:
                                    json.dump(annotations, f, indent=2)

                            # --- Polygon → YOLO Segment .txt ---
                            elif "Polygon" in label_mode:
                                label_name = img_name.replace(".jpg", "_seg.txt")
                                lines = []
                                for _, row in objects.iterrows():
                                    raw_path = row.get("path", [])
                                    points = [
                                        f"{round(float(cmd[1])/CANVAS_W, 6)} {round(float(cmd[2])/CANVAS_H, 6)}"
                                        for cmd in raw_path
                                        if len(cmd) >= 3 and cmd[0] in ("M", "L")
                                    ]
                                    if points:
                                        lines.append(f"{cid} " + " ".join(points))
                                with open(os.path.join(LABEL_DIR, label_name), "w") as f:
                                    f.write("\n".join(lines))

                            st.toast("บันทึกเรียบร้อย 🎉")

                        except Exception as e:
                            st.error(f"❌ บันทึกไม่สำเร็จ: {e}")