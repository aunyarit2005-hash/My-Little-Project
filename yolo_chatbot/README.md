# BreastVision Lab — Streamlit ultrasound research prototype

เว็บอัปโหลดภาพอัลตราซาวด์ แสดงผล YOLO รายกรอบ และค้นบทสรุปวิชาการประกอบคำตอบภาษาไทย ใช้ได้โดยไม่ต้องมี API key

## ความสามารถ

- โมเดล `best.pt`: YOLO11n detection, `benign` / `malignant` ประมวลผลบน CPU
- เลือกกรอบ เปิด–ปิด Bounding box และถามคะแนน/ตำแหน่งของกรอบนั้นจากผลจริง
- ค้นด้วย TF-IDF character n-grams จาก 5 บทสรุปใน `rag.py` ซึ่งมาจาก 3 แหล่งข้อมูล ไม่ได้ค้นเว็บสดหรืออ่านเอกสารทั้งฉบับ
- LLM เป็นทางเลือกสำหรับเรียบเรียงหลักฐาน มี fallback เมื่อเชื่อมต่อไม่ได้หรือรูปแบบอ้างอิงไม่ผ่าน
- ดาวน์โหลดภาพ PNG, ผล JSON และรายงาน Markdown พร้อม image/model hash และเวลาประมวลผล
- เปลี่ยนภาพหรือ threshold จะล้างผลและแชตเดิม ต้องประมวลผลใหม่
- รองรับผล `segment` หากผู้ดูแลเปลี่ยนเป็นโมเดล segmentation ที่ฝึกแล้วและมีคลาสเดียวกัน แต่โมเดลที่รวมมา **ไม่มี mask** จึงไม่แสดงค่ารูปร่างปลอม

## รันในเครื่อง

ใช้ Python 3.12 และรันจาก root ของ repository เพื่อให้ Streamlit อ่าน `.streamlit/config.toml`:

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1` หรือ macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r yolo_chatbot/requirements.txt
python -m streamlit run yolo_chatbot/app.py
```

Requirements ใช้ CPU wheels ของ PyTorch เพื่อลดการดาวน์โหลด CUDA เหมาะกับ Streamlit Cloud บน Linux x86-64 สำหรับ macOS ให้ติดตั้ง torch/torchvision ที่รองรับระบบนั้นแยกจากรายการ `+cpu`

## Deploy บน Streamlit Community Cloud

1. ลงชื่อเข้าใช้ https://share.streamlit.io/ และให้บัญชี Streamlit เข้าถึง repository
2. เลือก **Create app → Deploy a public app from GitHub**
3. Repository: `aunyarit2005-hash/My-Little-Project`
4. Branch: `main` (หรือ branch ที่มีการแก้ไขครบแล้ว)
5. Main file path: `yolo_chatbot/app.py`
6. Advanced settings: เลือก Python **3.12**; เวอร์ชันแรกไม่ต้องใส่ Secrets
7. กด Deploy แล้วตรวจ build logs และทดลองอัปโหลดภาพที่ลบข้อมูลระบุตัวบุคคลแล้ว

ไฟล์ `packages.txt` อยู่ root สำหรับไลบรารีระบบของ OpenCV ส่วน Python dependencies อยู่ `yolo_chatbot/requirements.txt` และ `best.pt` ต้องอยู่ข้าง `app.py`

## LLM ทางเลือก

คัดลอกตัวอย่างจาก `secrets.example.toml` ไป **App settings → Secrets** บน Streamlit Cloud หรือ `.streamlit/secrets.toml` ในเครื่อง ห้าม commit key จริง

- ค่าเริ่มต้น `provider = "none"`: ตอบค่าจากภาพและค้นหลักฐานโดยไม่เรียกบริการ LLM
- `provider = "openai"`: ตั้ง `model`, `api_key` และ HTTPS `base_url` สำหรับ Chat Completions ที่รองรับ JSON mode (`json_object`, `max_completion_tokens`, `store`)
- `provider = "ollama"`: ตั้ง `model`, `base_url` และ token ถ้าบริการนั้นกำหนด ต้องมี Ollama server ที่เครื่อง deploy เข้าถึงได้จริง `127.0.0.1` บน Cloud ไม่ใช่คอมพิวเตอร์ผู้ใช้
- ใช้ environment variables `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` แทน Secrets ได้
- การใช้บริการ LLM อาจมีค่าใช้จ่ายตามผู้ให้บริการ จึงปิดไว้เป็นค่าเริ่มต้น

เมื่อผู้ใช้เปิด LLM ระบบส่งเฉพาะคำถามและบทสรุปที่ค้นได้ ไม่ส่งภาพ ผล YOLO หรือประวัติแชต ค่าจากภาพตอบด้วย Python แยกจากข้อความ LLM ประวัติแชตใช้แสดงผล ไม่ได้เป็นบริบทสนทนาอิสระ

ตรวจรหัสอ้างอิงและรูปแบบคำตอบก่อนแสดง แต่ไม่ได้พิสูจน์ว่าข้อความ LLM ทุกประโยคมีหลักฐานรองรับ จึงยังต้องตรวจเทียบเอกสารต้นทาง

## ขอบเขตและข้อมูล

ผลนี้เป็นต้นแบบการศึกษา ไม่ใช่การวินิจฉัย คลาส benign/malignant และ confidence ไม่ยืนยันสถานะโรค ไม่มีกรอบไม่ได้แปลว่าไม่มีโรค ข้อความต้นแบบยังไม่ได้รับการตรวจทานจากแพทย์

บน Cloud ภาพจะถูกส่งไปเซิร์ฟเวอร์ที่รันแอป โค้ดไม่บันทึกภาพหรือแชตลงดิสก์โดยเจตนา แต่ข้อมูลอยู่ในหน่วยความจำของ session และระบบ hosting จัดการวงจรชีวิตของข้อมูล ใช้เฉพาะภาพที่ลบข้อมูลระบุตัวบุคคลแล้ว ไม่มีระบบตรวจชนิดภาพหรือข้อมูลระบุตัวบุคคลอัตโนมัติ

ไฟล์ .pt เป็น checkpoint ของผู้ดูแลที่กำหนดไว้ ไม่เปิดให้ผู้ใช้เว็บอัปโหลดโมเดลอื่น การเพิ่ม segmentation ต้องใช้โมเดลที่ฝึกแล้ว ค่ารูปร่างคำนวณจาก contour ที่ทำนายและรายงานเป็นพิกเซล ไม่แปลงเป็นมิลลิเมตรหรือ BI-RADS

## ตรวจสอบ

```bash
cd yolo_chatbot
python -m unittest test_chat.py test_cloud.py test_app.py -v
```

`test_app.py` ใช้ Streamlit AppTest และภาพสังเคราะห์ทดสอบ checkpoint จริง ไม่ใช่ชุดประเมินทางคลินิก ดูสถานะการตรวจล่าสุดใน `VALIDATION.md`

## เอกสารการติดตั้ง

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- https://docs.ultralytics.com/tasks/segment/
- https://developers.openai.com/api/docs/guides/structured-outputs


### หน้าจอแบบ prototype (UI 2026.09.15)

จัดเป็นสี่การ์ด: ภาพและขอบเขต, ลักษณะรอยโรค, แชต, เอกสารประกอบ โทนเทาเข้มและเขียวมิ้นต์ ภาพรักษาอัตราส่วนและแชตเลื่อนภายในกรอบ อัปโหลดและดาวน์โหลดอยู่ในแถบพับได้

ใช้ผลจริงจากโมเดล: อัตราส่วนกว้างต่อสูงมาจาก BB ส่วนพื้นที่ ความยาวขอบ และ Circularity แสดงเฉพาะเมื่อมี mask จริง ไม่มีการเติมค่าจำลองจาก prototype ในภาพผู้ใช้
