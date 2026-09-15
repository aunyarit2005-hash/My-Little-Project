import hashlib
import json
import threading
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image, UnidentifiedImageError
from chat import NOTICE, answer
from image_analysis import read_image, prediction_data, render_overlay, image_png, make_report
from llm import LLMConfig, load_config
from rag import KNOWLEDGE, rag_answer
from ui import apply_ui

st.set_page_config(page_title="BreastVision Lab", page_icon="🔬", layout="wide")
apply_ui()
st.title("สำรวจรอยโรคในภาพ")
st.caption("วิเคราะห์ภาพอัลตราซาวด์ · สำรวจผลทีละกรอบ · อ่านคำอธิบายพร้อมแหล่งอ้างอิง")
st.caption(NOTICE)


@st.cache_resource
def load_model():
    from ultralytics import YOLO
    path = Path(__file__).with_name("best.pt")
    if not path.exists():
        raise FileNotFoundError("Missing best.pt")
    model = YOLO(str(path))
    if model.task not in {"detect", "segment"} or set(model.names.values()) != {"benign", "malignant"}:
        raise ValueError("Expected benign/malignant detection or segmentation model")
    return model, threading.Lock(), hashlib.sha256(path.read_bytes()).hexdigest()


def clear_result():
    for name in ("result", "messages", "selected_frame", "pending_question"):
        st.session_state.pop(name, None)


try:
    settings = st.secrets.get("llm", {})
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    settings = {}
try:
    llm_config = load_config(settings)
except ValueError:
    llm_config = LLMConfig()
    st.info("การตั้งค่า LLM ยังไม่พร้อม ระบบค้นบทสรุปและตอบผลภาพได้ตามปกติ")

with st.sidebar:
    st.header("การตั้งค่า")
    threshold = st.slider("เกณฑ์กรอง confidence", 0.05, 0.95, 0.50, 0.05)
    st.caption("กรองกรอบที่จะแสดง ไม่ใช่เกณฑ์วินิจฉัย")
    use_llm = False
    if llm_config.enabled:
        use_llm = st.checkbox("ใช้ LLM เรียบเรียงหลักฐาน", value=False)
        st.caption("เมื่อเปิด จะส่งคำถามและบทสรุปเอกสารไปยังบริการ LLM ที่ผู้ดูแลตั้งค่า ไม่ส่งภาพ ผลตรวจจับ หรือประวัติแชต กรุณาไม่ใส่ข้อมูลระบุตัวบุคคลในคำถาม")
    else:
        st.caption("ค้นความรู้จากบทสรุปที่คัดไว้ · ใช้งานได้โดยไม่เรียก LLM")
    if st.button("ล้างภาพและบทสนทนา"):
        st.session_state.clear()
        st.session_state["upload_key"] = uuid4().hex
        st.rerun()
    with st.expander("ฐานความรู้และขอบเขต"):
        st.write("ค้นด้วย TF-IDF จาก 5 บทสรุปที่คัดไว้ ไม่ได้ค้นอินเทอร์เน็ตสดหรืออ่าน PDF ทั้งฉบับ")
        for item in KNOWLEDGE:
            st.markdown(f"- [{item['title']}]({item['url']})")
        st.caption("แหล่งอ้างอิงช่วยอธิบายความรู้ ไม่ได้ยืนยันความแม่นยำของ best.pt")

with st.expander("อัปโหลดภาพและประมวลผล", expanded=not st.session_state.get("result")):
    st.caption("ภาพจะถูกส่งไปประมวลผลบนเซิร์ฟเวอร์ที่รันแอป โค้ดนี้ไม่บันทึกภาพลงดิสก์ ใช้ภาพที่นำข้อมูลระบุตัวผู้ป่วยออกแล้วเท่านั้น")
    uploaded = st.file_uploader("เลือกภาพ PNG หรือ JPG (ไม่เกิน 10 MB)", type=["png", "jpg", "jpeg"], key=st.session_state.get("upload_key", "upload"))
    raw = uploaded.getvalue() if uploaded else None
    input_key = (hashlib.sha256(raw).hexdigest(), threshold) if raw else None
    if st.session_state.get("input_key") != input_key:
        clear_result()
        st.session_state.pop("confirmed", None)
        st.session_state["input_key"] = input_key
    picture = None
    if raw:
        try:
            picture = read_image(raw)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.error("อ่านภาพไม่ได้: กรุณาใช้ PNG/JPG ไม่เกิน 10 MB และ 20 ล้านพิกเซล ภาพต้องไม่เป็นสีเดียวทั้งหมด")
            st.stop()
    
    confirmed = st.checkbox("ยืนยันว่าเป็นภาพอัลตราซาวด์สำหรับโมเดลนี้ และนำข้อมูลระบุตัวผู้ป่วยออกแล้ว", key="confirmed")
    if st.button("ประมวลผลภาพ", type="primary", disabled=not confirmed or picture is None):
        clear_result()
        try:
            with st.spinner("กำลังประมวลผลด้วย YOLO…"):
                model, lock, model_hash = load_model()
                with lock:
                    prediction = model.predict(picture, conf=threshold, imgsz=640, device="cpu", verbose=False)[0]
                    data = prediction_data(prediction, input_key[0], threshold, model_hash, model.task)
                st.session_state["result"] = data
                st.session_state["messages"] = [{"role": "assistant", "content": answer("summary", data)}]
            st.rerun()
        except Exception:
            st.error("ประมวลผลไม่สำเร็จ กรุณาลองใหม่ หรือติดต่อผู้ดูแลเพื่อตรวจโมเดลและ dependencies")

data = st.session_state.get("result")
ds = data["detections"] if data else []
left, right = st.columns([1.5, 1], gap="medium")
selected = 0
with left, st.container(border=True, key="image_panel"):
    heading, boxes, masks = st.columns([2, 1, 1])
    heading.subheader("ภาพและขอบเขต")
    show_boxes = boxes.checkbox("Bounding box", value=True)
    has_masks = any(d.get("polygon") for d in ds)
    show_masks = masks.checkbox("Mask", value=has_masks, disabled=not has_masks)
    st.divider()
    if ds:
        selected = st.selectbox("ตำแหน่งที่เลือก", list(range(len(ds) + 1)),
                                index=1, format_func=lambda i: "ทุกกรอบ" if i == 0 else f"กรอบ {i:02d}", key="selected_frame")
    if picture is not None:
        st.caption(f"ULTRASOUND · {picture.width} × {picture.height} px")
        shown = render_overlay(picture, ds, show_boxes, show_masks, selected or None)
        st.image(shown, width="stretch")
    else:
        st.markdown('<div class="image-placeholder">อัปโหลดภาพอัลตราซาวด์<span>PNG / JPG · สูงสุด 10 MB</span></div>', unsafe_allow_html=True)
    st.caption(f"ตรวจพบ {len(ds)} กรอบ · เส้นขอบ = Bounding box · พื้นสี = Mask" if data else "เลือกภาพจากแถบอัปโหลดด้านบน แล้วกดประมวลผลภาพ")

with right, st.container(border=True, key="features_panel"):
    st.subheader("ลักษณะรอยโรค")
    st.caption("ค่าจากผลโมเดลของภาพที่เลือก")
    st.divider()
    item = ds[selected - 1] if selected else None
    a, b = st.columns(2)
    a.markdown("ตำแหน่งที่เลือก")
    a.markdown(f"**กรอบ {selected:02d}**" if selected else "**ทุกกรอบ**" if ds else "**—**")
    b.markdown("ผลจำแนกจากโมเดล")
    b.markdown(f"**{item['label']}**" if item else "**—**")
    st.divider()
    shape = (item.get("shape") or {}) if item else {}
    ratio = None
    if item:
        x1, y1, x2, y2 = item["bbox"]
        ratio = (x2 - x1) / (y2 - y1) if y2 > y1 else None
    a, b = st.columns(2)
    a.metric("พื้นที่ mask / px²", f"{shape['area_px2']:,.1f}" if shape else "—")
    b.metric("ความยาวขอบ / px", f"{shape['perimeter_px']:,.1f}" if shape else "—")
    a, b = st.columns(2)
    a.metric("Circularity", f"{shape['circularity']:.3f}" if shape else "—")
    b.metric("BB กว้าง : สูง", f"{ratio:.2f}" if ratio is not None else "—")
    st.divider()
    if item:
        st.markdown(f"**Confidence {item['confidence']:.3f}** · คะแนนของโมเดล ไม่ใช่โอกาสเป็นโรค")
    elif data and not ds:
        st.write("ไม่พบกรอบที่ผ่านเกณฑ์ ไม่ได้ยืนยันว่าไม่มีโรค")
    else:
        st.caption("เลือกกรอบเพื่อดูรายละเอียดเฉพาะตำแหน่ง")
    if not has_masks:
        st.caption("ยังไม่มี mask จากโมเดล จึงไม่มีค่าพื้นที่ ความยาวขอบ และ Circularity ของรอยโรค")
    st.caption("ไม่มีสเกลสำหรับแปลงเป็นมิลลิเมตร · ไม่ประเมินความเสี่ยงมะเร็งจากรูปร่างเพียงอย่างเดียว")

left, right = st.columns([1.5, 1], gap="medium")
with left, st.container(border=True, key="chat_panel"):
    st.subheader("ถามเกี่ยวกับผลวิเคราะห์")
    st.caption("คำตอบอ้างอิงกรอบที่เลือกและบทสรุปเอกสาร")
    st.divider()
    quick = st.columns(3)
    for col, label, question in zip(quick, ["อธิบายรูปร่าง", "Mask ต่างจาก BB?", "ดูคะแนน"],
                                   ["อธิบายรูปร่างของกรอบที่เลือก", "Mask ต่างจาก bounding box อย่างไร", "คะแนนกรอบที่ตรวจพบเท่าไร"]):
        if col.button(label, key="quick_" + label, disabled=not data, width="stretch"):
            st.session_state["pending_question"] = question
    with st.container(height=310, border=False):
        if not data:
            st.info("ประมวลผลภาพเพื่อเริ่มถามเกี่ยวกับผลวิเคราะห์")
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])
    typed = st.chat_input("ลองถาม: mask คืออะไร", max_chars=1000, disabled=not data)
    pending = st.session_state.pop("pending_question", None)
    question = typed or pending
    if question and data:
        with st.spinner("กำลังค้นคำอธิบาย…"):
            reply = rag_answer(question, data, llm_config if use_llm else LLMConfig(), selected=selected or None)
        scope = f" (กรอบที่เลือก: {selected})" if selected else ""
        st.session_state["messages"].extend([{"role": "user", "content": question + scope}, {"role": "assistant", "content": reply}])
        st.session_state["messages"] = st.session_state["messages"][-30:]
        st.rerun()

with right, st.container(border=True, key="sources_panel"):
    st.subheader("เอกสารประกอบ")
    st.caption("เปิดอ่านแหล่งข้อมูลต้นทาง")
    st.divider()
    seen = set()
    for source in KNOWLEDGE:
        if source["url"] in seen:
            continue
        seen.add(source["url"])
        st.markdown(f"**[{source['title']} ↗]({source['url']})**")
        st.caption(source["kind"])
        st.divider()
    st.caption("เอกสารอธิบายหลักการ ไม่ได้ยืนยันผลทำนายของภาพนี้")

if data:
    with st.expander("ข้อมูลตรวจจับและดาวน์โหลด"):
        rows = [{"กรอบ": i, "คลาส": d["label"], "confidence": round(d["confidence"], 4),
                 "bbox (px)": ", ".join(f"{v:.1f}" for v in d["bbox"])} for i, d in enumerate(ds, 1)]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        a, b, c = st.columns(3)
        a.download_button("ภาพพร้อมผล", image_png(shown), "prediction.png", "image/png")
        b.download_button("ข้อมูล JSON", json.dumps(data, ensure_ascii=False, indent=2), "prediction.json", "application/json")
        c.download_button("รายงาน Markdown", make_report(data), "prediction-report.md", "text/markdown")
st.caption("BreastVision · UI 2026.09.15 · ต้นแบบเพื่อการศึกษา · ยังไม่ผ่านการประเมินทางคลินิก")
