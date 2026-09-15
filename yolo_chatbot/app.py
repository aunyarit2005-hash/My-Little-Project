import hashlib
import json
import threading
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image, UnidentifiedImageError
from chat import NOTICE, answer, next_steps
from image_analysis import read_image, prediction_data, render_overlay, image_png, make_report
from llm import LLMConfig, load_config
from rag import KNOWLEDGE, rag_answer
from ui import apply_ui

st.set_page_config(page_title="BreastVision Lab", page_icon="🔬", layout="wide")
apply_ui()
st.title("BreastVision Lab")
st.caption("วิเคราะห์ภาพอัลตราซาวด์ · สำรวจผลทีละกรอบ · อ่านคำอธิบายพร้อมแหล่งอ้างอิง")
st.warning(NOTICE)


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
    show_boxes = st.checkbox("แสดง Bounding box", value=True)
    show_masks = st.checkbox("แสดง Mask เมื่อโมเดลรองรับ", value=True)
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

st.caption("ภาพจะถูกส่งไปประมวลผลบนเซิร์ฟเวอร์ที่รันแอป โค้ดนี้ไม่บันทึกภาพลงดิสก์ ใช้ภาพที่นำข้อมูลระบุตัวผู้ป่วยออกแล้วเท่านั้น")
uploaded = st.file_uploader("เลือกภาพ PNG หรือ JPG (ไม่เกิน 10 MB)", type=["png", "jpg", "jpeg"], key=st.session_state.get("upload_key", "upload"))
raw = uploaded.getvalue() if uploaded else None
input_key = (hashlib.sha256(raw).hexdigest(), threshold) if raw else None
if st.session_state.get("input_key") != input_key:
    clear_result()
    st.session_state.pop("confirmed", None)
    st.session_state["input_key"] = input_key
if not raw:
    st.info("เริ่มจากอัปโหลดภาพ จากนั้นเลือกกรอบเพื่อดูคะแนน ตำแหน่ง และถามคำอธิบายผล")
    st.stop()
try:
    picture = read_image(raw)
except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
    st.error("อ่านภาพไม่ได้: กรุณาใช้ PNG/JPG ไม่เกิน 10 MB และ 20 ล้านพิกเซล ภาพต้องไม่เป็นสีเดียวทั้งหมด")
    st.stop()

confirmed = st.checkbox("ยืนยันว่าเป็นภาพอัลตราซาวด์สำหรับโมเดลนี้ และนำข้อมูลระบุตัวผู้ป่วยออกแล้ว", key="confirmed")
if st.button("ประมวลผลภาพ", type="primary", disabled=not confirmed):
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
left, right = st.columns([1.15, 1])
with left:
    st.subheader("ภาพและผลตรวจจับ")
    selected = 0
    if data and data["detections"]:
        selected = st.selectbox("เลือกกรอบ", list(range(len(data["detections"]) + 1)),
                                format_func=lambda i: "ทุกกรอบ" if i == 0 else f"กรอบ {i}", key="selected_frame")
    shown = render_overlay(picture, data["detections"], show_boxes, show_masks, selected or None) if data else picture
    st.image(shown, width="stretch")
    if data:
        if data["task"] == "detect":
            st.caption("โมเดลนี้ให้เฉพาะ Bounding box จึงยังไม่มี mask หรือค่ารูปร่างของรอยโรค")
        if selected:
            item = data["detections"][selected - 1]
            st.markdown(f"**กรอบ {selected} · {item['label']} · confidence {item['confidence']:.3f}**")
            if item.get("shape"):
                s = item["shape"]
                st.write({"พื้นที่จาก contour (px²)": s["area_px2"], "ความยาวขอบ (px)": s["perimeter_px"], "Circularity": s["circularity"]})
                st.caption("คำนวณจาก contour ที่โมเดลทำนาย ไม่มีสเกลมิลลิเมตร และไม่ใช่คะแนน BI-RADS")
        with st.expander("ข้อมูลตรวจจับและดาวน์โหลด"):
            rows = [{"กรอบ": i, "คลาส": d["label"], "confidence": round(d["confidence"], 4),
                     "bbox (px)": ", ".join(f"{v:.1f}" for v in d["bbox"])} for i, d in enumerate(data["detections"], 1)]
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.write("ไม่พบกรอบที่ผ่านเกณฑ์ ไม่ได้ยืนยันว่าไม่มีโรค")
            st.download_button("ภาพพร้อมผล", image_png(shown), "prediction.png", "image/png")
            st.download_button("ข้อมูล JSON", json.dumps(data, ensure_ascii=False, indent=2), "prediction.json", "application/json")
            st.download_button("รายงาน Markdown", make_report(data), "prediction-report.md", "text/markdown")
with right:
    st.subheader("ผู้ช่วยอธิบายผล")
    if not data:
        st.info("กดประมวลผลภาพก่อนเริ่มแชต")
    else:
        ds = data["detections"]
        a, b, c = st.columns(3)
        a.metric("กรอบทั้งหมด", len(ds))
        b.metric("คลาส benign", sum(d["label"] == "benign" for d in ds))
        c.metric("คลาส malignant", sum(d["label"] == "malignant" for d in ds))
        st.caption("จำนวนกรอบจากโมเดล ไม่ใช่จำนวนรอยโรคที่ยืนยันแล้ว")
        st.info(next_steps(data))
        st.caption("คำถามผลภาพจะอ้างถึงกรอบที่เลือก หรือระบุ เช่น ‘คะแนนกรอบที่ 1 เท่าไร’")
        quick = st.columns(3)
        for col, label, question in zip(quick, ["สรุปผล", "ดูคะแนน", "ขั้นตอนถัดไป"], ["สรุปผล", "คะแนนกรอบที่ตรวจพบเท่าไร", "ควรทำอย่างไรต่อไป"]):
            if col.button(label, key="quick_" + label):
                st.session_state["pending_question"] = question
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])
        typed = st.chat_input("ถามเกี่ยวกับผลหรือเอกสาร", max_chars=1000)
        pending = st.session_state.pop("pending_question", None)
        question = typed or pending
        if question:
            with st.spinner("กำลังค้นคำอธิบาย…"):
                reply = rag_answer(question, data, llm_config if use_llm else LLMConfig(), selected=selected or None)
            scope = f" (กรอบที่เลือก: {selected})" if selected else ""
            st.session_state["messages"].extend([{"role": "user", "content": question + scope}, {"role": "assistant", "content": reply}])
            st.session_state["messages"] = st.session_state["messages"][-30:]
            st.rerun()
st.caption("ต้นแบบเพื่อการศึกษา · บทสรุปอ้างอิงคัดไว้วันที่ 14 ก.ย. 2026 · ยังไม่ผ่านการประเมินทางคลินิก")
