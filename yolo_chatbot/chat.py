"""Deterministic answers about the actual image; no patient data sent to an LLM."""
import re

NOTICE = 'ผลนี้เป็นการทำนายของโมเดลสำหรับต้นแบบการศึกษา ไม่ใช่การวินิจฉัย และ confidence ไม่ใช่โอกาสเป็นโรค'
INTENTS = ['summary', 'confidence', 'next', 'prepare', 'limits', 'knowledge']

def select_intent(question, model=''):
    # Retained for callers of the original API. Intent selection is always local.
    return rule_intent(question.lower()), ''

def rule_intent(q):
    if any(w in q for w in ['รักษา', 'ยาอะไร', 'เป็นมะเร็งไหม', 'หายไหม', 'ระยะไหน', 'ปกติไหม', 'diagnose me', 'treatment', 'medicine', 'do i have cancer']):
        return 'limits'
    if any(w in q for w in ['เตรียม', 'prepare', 'เอาอะไร']):
        return 'prepare'
    if any(w in q for w in ['มั่นใจ', 'confidence', 'คะแนน', 'เปอร์เซ็นต์']):
        return 'confidence'
    if any(w in q for w in ['ต่อไป', 'ทำยังไง', 'ทำอย่างไร', 'แนะนำ', 'next']):
        return 'next'
    if any(w in q for w in ['อัลตราซาว', 'ultrasound', 'ชิ้นเนื้อ', 'biopsy', 'mask', 'รูปร่าง', 'segmentation']):
        return 'knowledge'
    if any(w in q for w in ['สรุป', 'พบ', 'กี่', 'ผล', 'summary', 'count', 'benign', 'malignant', 'กรอบ', 'ตำแหน่ง', 'box', 'bbox']):
        return 'summary'
    return 'limits'


def select_frame(question, selected=None):
    q = question.translate(str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789'))
    match = re.search(r'(?:กรอบ|ตำแหน่ง|box|lesion)\s*(?:ที่\s*)?#?\s*(\d+)', q, re.I)
    return int(match.group(1)) if match else selected


def detection_answer(question, data, selected=None):
    """Answer score/position requests from detections, never from retrieved text."""
    frame = select_frame(question, selected)
    detections = data['detections']
    if frame is not None:
        if not 1 <= frame <= len(detections):
            return f'ไม่พบกรอบที่ {frame} ในผลภาพนี้ (มีทั้งหมด {len(detections)} กรอบ)'
        chosen = [(frame, detections[frame - 1])]
    else:
        chosen = list(enumerate(detections, 1))
    if not chosen:
        return summary(data)
    lines = []
    for i, d in chosen:
        coords = ', '.join(f'{v:.1f}' for v in d['bbox'])
        lines.append(f"- กรอบ {i}: คลาส {d['label']} · confidence {d['confidence']:.3f} · bbox ({coords}) px")
    return '\n'.join(lines)

def summary(data):
    ds = data['detections']
    if not ds:
        return 'โมเดลตรวจไม่พบกรอบที่ผ่านเกณฑ์ที่ตั้งไว้ จึงยังสรุปไม่ได้ และไม่ได้หมายความว่าภาพปกติ'
    lines = [f'โมเดลตรวจพบ {len(ds)} กรอบ (จำนวนกรอบไม่จำเป็นต้องเท่ากับจำนวนรอยโรคจริง)']
    for i, d in enumerate(ds, 1):
        lines.append(f"- กรอบ {i}: คลาส {d['label']} · confidence {d['confidence']:.3f}")
    return '\n'.join(lines)

def next_steps(data):
    labels = {d['label'] for d in data['detections']}
    if 'malignant' in labels:
        return 'โมเดลให้คลาส malignant อย่างน้อยหนึ่งกรอบ แต่ยังยืนยันมะเร็งไม่ได้ ควรติดต่อแพทย์ผู้ส่งตรวจเพื่อนำภาพและรายงานจริงมาทบทวน และให้แพทย์พิจารณาการตรวจเพิ่มเติม'
    if labels == {'benign'}:
        return 'แม้โมเดลให้คลาส benign ก็ยังยืนยันว่าไม่มีโรคไม่ได้ ควรให้แพทย์ประเมินภาพและรายงานจริง และทำตามนัดเดิม ไม่เปลี่ยนแผนดูแลจากผล AI'
    return 'ระบบยังให้ข้อสรุปไม่ได้ ควรตรวจสอบว่าภาพตรงกับชนิดข้อมูลที่โมเดลฝึกและมีคุณภาพเพียงพอ แล้วให้ผู้เชี่ยวชาญประเมิน อย่าใช้การปรับ threshold เพื่อยืนยันว่าปกติ'

def answer(intent, data):
    content = {
        'summary': lambda: summary(data),
        'next': lambda: next_steps(data),
        'confidence': lambda: 'confidence เป็นคะแนนของโมเดลสำหรับแต่ละกรอบ ไม่ใช่เปอร์เซ็นต์โอกาสเป็นมะเร็ง ไม่ใช่ความแม่นยำของโมเดลโดยรวม และใช้กำหนดความเร่งด่วนไม่ได้',
        'prepare': lambda: 'เตรียมภาพและรายงานการตรวจเดิม จดอาการและเวลาที่เริ่มเป็น พร้อมคำถาม เช่น ผลตรวจจริงหมายถึงอะไร ต้องตรวจเพิ่มเติมหรือไม่ และควรติดตามตามแผนใด โดยให้แพทย์เป็นผู้กำหนด',
        'limits': lambda: 'บอตนี้ช่วยสรุปกรอบตรวจจับ อธิบาย confidence แนะนำการนำผลไปปรึกษาแพทย์ และเตรียมคำถามได้ แต่ไม่สามารถยืนยันโรค ระยะโรค เลือกยา หรือกำหนดการรักษาจากผล YOLO ได้ ลองถาม “สรุปผล” หรือ “ควรทำอย่างไรต่อไป”'
    }.get(intent, lambda: 'ระบบยังตอบคำถามนี้ไม่ได้')()
    return content + '\n\n' + NOTICE
