import hashlib
import io
import json
import threading
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from chat import NOTICE, answer, next_steps, select_intent

st.set_page_config(page_title='YOLO • ผู้ช่วยอธิบายผล', page_icon='🔬', layout='wide')
st.title('🔬 ผู้ช่วยอธิบายผล YOLO')
st.caption('อัปโหลดภาพ • ตรวจจับบริเวณ • ถามเกี่ยวกับผล')
st.warning(NOTICE)
st.caption('คำแนะนำเป็นข้อความต้นแบบ ยังไม่ได้รับการตรวจทานจากแพทย์ ไม่ใช้ตัดสินใจทางคลินิก')

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    path = Path(__file__).with_name('best.pt')
    if not path.exists():
        raise FileNotFoundError('วาง best.pt ในโฟลเดอร์เดียวกับ app.py')
    model = YOLO(str(path))
    if model.task != 'detect' or set(model.names.values()) != {'benign', 'malignant'}:
        raise ValueError('โมเดลต้องเป็น detection ที่มีคลาส benign และ malignant')
    return model, threading.Lock()

with st.sidebar:
    st.header('การตั้งค่า')
    threshold = st.slider('เกณฑ์กรอง confidence', 0.05, 0.95, 0.50, 0.05)
    st.caption('เป็นเกณฑ์แสดงกรอบ ไม่ใช่เกณฑ์วินิจฉัยโรค')
    use_llm = st.checkbox('ใช้ Ollama ช่วยเข้าใจคำถาม', value=False)
    llm_model = st.text_input('ชื่อโมเดลที่ติดตั้งใน Ollama', placeholder='ชื่อโมเดลในเครื่อง', disabled=not use_llm)
    if use_llm and not llm_model.strip():
        st.info('ยังไม่ได้ระบุชื่อโมเดล จึงใช้แชตตามกฎ')
    st.caption('โหมดพื้นฐานใช้กฎ ไม่มี API key; Ollama ใช้ localhost และเลือกหัวข้อคำตอบเท่านั้น')
    if st.button('ล้างภาพและบทสนทนา'):
        st.session_state.clear()
        st.session_state['upload_key'] = hashlib.sha256(__import__('os').urandom(16)).hexdigest()
        st.rerun()

uploaded = st.file_uploader('เลือกภาพ PNG หรือ JPG (ไม่เกิน 10 MB)', type=['png', 'jpg', 'jpeg'], key=st.session_state.get('upload_key', 'upload'))
raw = uploaded.getvalue() if uploaded else None
key = (hashlib.sha256(raw).hexdigest(), threshold) if raw else None
if st.session_state.get('input_key') != key:
    for item in ['result', 'annotated', 'messages']:
        st.session_state.pop(item, None)
    st.session_state['input_key'] = key
if not raw:
    st.info('อัปโหลดภาพอัลตราซาวด์ที่นำข้อมูลระบุตัวบุคคลออกแล้ว เพื่อเริ่มใช้งาน')
    st.stop()
if len(raw) > 10 * 1024 * 1024:
    st.error('ไฟล์ใหญ่เกิน 10 MB')
    st.stop()
try:
    with Image.open(io.BytesIO(raw)) as source:
        if source.width * source.height > 20_000_000:
            raise ValueError('ภาพต้องมีไม่เกิน 20 ล้านพิกเซล')
        picture = ImageOps.exif_transpose(source).convert('RGB')
        if all(low == high for low, high in picture.getextrema()):
            raise ValueError('ภาพมีสีเดียวทั้งหมด กรุณาใช้ภาพอัลตราซาวด์ที่มีข้อมูลภาพ')
except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
    st.error(f'อ่านภาพไม่ได้: {exc}')
    st.stop()
left, right = st.columns([1, 1])
with left:
    st.subheader('ภาพและผลตรวจจับ')
    st.image(st.session_state.get('annotated', picture), width='stretch')
    confirmed = st.checkbox('ยืนยันว่าภาพนี้เป็นภาพอัลตราซาวด์สำหรับโมเดลนี้ และนำข้อมูลระบุตัวบุคคลออกแล้ว')
    if st.button('ประมวลผลภาพ', type='primary', disabled=not confirmed):
        # Clear old results before any attempt, including failed attempts.
        for item in ['result', 'annotated', 'messages']:
            st.session_state.pop(item, None)
        try:
            with st.spinner('กำลังประมวลผลด้วย YOLO…'):
                model, lock = load_model()
                with lock:
                    prediction = model.predict(picture, conf=threshold, imgsz=640, device='cpu', verbose=False)[0]
                    detections = [{'label': prediction.names[int(b.cls.item())],
                                   'confidence': float(b.conf.item()),
                                   'bbox': b.xyxy[0].cpu().tolist()} for b in prediction.boxes]
                    plotted = prediction.plot()[:, :, ::-1].copy()
                st.session_state['result'] = {'image_sha256': key[0], 'threshold': threshold, 'detections': detections}
                st.session_state['annotated'] = plotted
                st.session_state['messages'] = [{'role': 'assistant', 'content': answer('summary', st.session_state['result'])}]
            st.rerun()
        except Exception as exc:
            st.error(f'ประมวลผลไม่สำเร็จ: {exc}')
            st.info('ตรวจสอบการติดตั้ง dependencies และไฟล์ best.pt ตาม README')
with right:
    st.subheader('ถามเกี่ยวกับผล')
    data = st.session_state.get('result')
    if not data:
        st.info('กดประมวลผลภาพก่อนเริ่มแชต')
    else:
        st.info(next_steps(data))
        with st.expander('ดูข้อมูลตรวจจับ'):
            st.json(data)
            st.download_button('ดาวน์โหลด JSON', json.dumps(data, ensure_ascii=False, indent=2), 'prediction.json', 'application/json')
        for message in st.session_state.get('messages', []):
            with st.chat_message(message['role']):
                st.write(message['content'])
        st.caption('ลองถาม: สรุปผล / confidence คืออะไร / ควรทำอย่างไรต่อไป / เตรียมอะไรไปพบแพทย์')
        question = st.chat_input('พิมพ์คำถามเกี่ยวกับภาพนี้', max_chars=1000)
        if question:
            with st.spinner('กำลังตอบ…'):
                intent, warning = select_intent(question, llm_model.strip() if use_llm else '')
                reply = answer(intent, data)
                if warning:
                    reply = warning + '\n\n' + reply
                st.session_state['messages'].extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': reply}])
                st.session_state['messages'] = st.session_state['messages'][-30:]
            st.rerun()
st.caption('ประมวลผลในเครื่องที่รันแอป • ไม่บันทึกภาพลงดิสก์ • คำแนะนำใช้ข้อความที่กำหนดไว้ล่วงหน้า')
