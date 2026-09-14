# YOLO Ultrasound Chatbot — ต้นแบบการศึกษา

รวม best.pt ของผู้ใช้ (YOLO11n detection: benign/malignant), หน้า Streamlit, แชตภาษาไทยและคำแนะนำขั้นตอนถัดไป

## เริ่มใช้งานบน Windows / macOS / Linux
ติดตั้ง Python 3.11 หรือ 3.12 แล้วแตก ZIP เปิด terminal ในโฟลเดอร์นี้:

```bash
python -m venv .venv
```
Windows PowerShell: `.venv\Scripts\Activate.ps1`
Windows CMD: `.venv\Scripts\activate.bat`
macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```
เปิด http://localhost:8501 อัปโหลดภาพ PNG/JPG ยืนยันชนิดภาพ แล้วกดประมวลผล
การติดตั้ง PyTorch ที่ Ultralytics ใช้มีขนาดใหญ่ และต้องเชื่อมต่ออินเทอร์เน็ตตอนติดตั้ง
หากไม่พบ Ultralytics เวอร์ชันที่กำหนด ให้ใช้ environment เดิมที่เทรนไฟล์นี้ (metadata: 8.4.142) ไม่แก้ไฟล์ checkpoint เพื่อข้ามความเข้ากันได้

## การทำงาน
ภาพ → YOLO → JSON และภาพพร้อมกรอบ → เลือกหัวข้อคำถาม → คำตอบที่กำหนดไว้
ผลและแชตเก็บใน session; เปลี่ยนภาพหรือ threshold จะล้างผลเก่า ต้องประมวลผลใหม่
จำนวนกรอบไม่ใช่จำนวนรอยโรคที่ยืนยันแล้ว ไม่มีระบบตรวจคุณภาพ/ตรวจชนิดภาพอัตโนมัติ

## LLM ทางเลือก
ค่าเริ่มต้นเป็นแชตตามกฎ ใช้ได้โดยไม่มี API key ไม่ใช่ LLM อิสระ
หากมี Ollama ทำงานในเครื่องที่ http://127.0.0.1:11434 และติดตั้งโมเดลภาษาไว้แล้ว:
1. เปิด “ใช้ Ollama ช่วยเข้าใจคำถาม”
2. กรอกชื่อโมเดลตามที่ติดตั้ง
3. LLM รับเฉพาะคำถามเพื่อเลือก intent; Python สร้างคำตอบจากผลจริงและข้อความคงที่
ไม่ส่งภาพหรือ JSON ไปยัง LLM และไม่ให้ LLM แต่งคำแนะนำทางการแพทย์
หากเชื่อมต่อไม่สำเร็จ ระบบกลับไปใช้กฎ และแจ้งในคำตอบ
ไม่มี RAG หรือการสนทนาอิสระหลายหัวข้อในรุ่นนี้ ประวัติแชตใช้แสดงผล ไม่ส่งเข้า LLM

## ขอบเขต
- สำหรับการศึกษาเท่านั้น ยังไม่ผ่านการทวนสอบทางคลินิก
- ข้อความคำแนะนำยังไม่ได้รับการตรวจทานจากแพทย์
- benign ไม่ยืนยันว่าปลอดโรค, malignant ไม่ยืนยันมะเร็ง, ไม่พบกรอบไม่ใช่ผลปกติ
- confidence ไม่ใช่โอกาสเป็นโรค ไม่ใช้ตัดสินความเร่งด่วน/การรักษา
- ไม่กำหนดยา ระยะโรค ระยะติดตาม หรือสั่งตรวจเพิ่มเติมเอง
- ภาพและข้อความอาจมีข้อมูลสุขภาพ: ใช้ภาพที่นำข้อมูลระบุตัวบุคคลออกแล้ว รันในเครื่อง ไม่เปิดพอร์ตสาธารณะ
- โหลดเฉพาะ checkpoint ที่เชื่อถือได้ ไฟล์ .pt อาจมีวัตถุ Python; แอปไม่เปิดให้ผู้ใช้อัปโหลดโมเดลอื่น

## แหล่งอ้างอิงสำหรับพัฒนาต่อ
- https://docs.ultralytics.com/modes/predict/
- https://docs.streamlit.io/develop/api-reference/chat/st.chat_input
- https://docs.ollama.com/api/chat
- https://www.cancer.org/cancer/types/breast-cancer/screening-tests-and-early-detection.html

## การตรวจสอบ
เรียก `python -m unittest test_chat.py` ทดสอบกฎด้วยข้อมูลสังเคราะห์
สถานะการตรวจจริงในสภาพแวดล้อมผู้สร้างดู VALIDATION.md
