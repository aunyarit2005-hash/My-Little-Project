# Fine-tune SAM สำหรับ mask ภาพอัลตราซาวด์เต้านม

ชุดเริ่มต้นสำหรับ **SAM 1 ViT-B**: freeze image/prompt encoders และฝึก mask decoder ด้วย BCE + Dice loss จากภาพ RGB และ binary masks ใช้ batch ทีละ 1 ภาพ/รอยโรค พร้อม early stopping ตาม validation Dice ไม่มีการฝึกแยก benign/malignant ด้วย SAM

**ยังไม่ได้ฝึกกับ dataset จริงของผู้ใช้** และไม่มีผลความแม่นยำทางคลินิก ผลตรวจสอบโค้ดใช้ภาพสังเคราะห์และส่วนประกอบ SAM จริงขนาดเล็ก; ไม่ใช่ผลทดสอบ ViT-B บน GPU/Colab เต็มรูปแบบ

## เริ่มด้วย Colab

เปิด `train_sam_colab.ipynb` ใน Google Colab แล้วเลือก Runtime → Change runtime type → GPU เตรียมไฟล์ใน Google Drive หรืออัปโหลดเข้ารันไทม์เอง ตั้งค่า DATA และ GROUPS แล้วรันทีละเซลล์ สำรอง outputs ก่อน Colab หมดอายุ ไม่รับประกันว่าบัญชีจะมี GPU ว่าง

## รูปแบบข้อมูล

เก็บโฟลเดอร์ benign, malignant, normal ใต้โฟลเดอร์ dataset แต่ชื่อคลาสไม่ได้ใช้เป็นเป้าหมายการฝึก SAM ตัวอย่างคู่ไฟล์:

- `benign/benign (1).png`
- `benign/benign (1)_mask.png`
- `benign/benign (1)_mask_1.png` (ถ้ามี mask เพิ่ม)

รองรับภาพ PNG/JPG/JPEG และ mask PNG ขนาดเท่าภาพ, grayscale/RGB ที่ช่องสีเท่ากัน, ค่า 0/1 หรือ 0/255 เท่านั้น สีดำ=พื้นหลัง สีขาว=รอยโรค ต้องมี mask ของทุกภาพ; mask หายไม่ถูกตีความเป็น normal

หลาย mask ต่อภาพจะรวมกันแล้วแยกด้วย connected components (8-connectivity) หนึ่ง component ต่อหนึ่ง box prompt ต้องตรวจข้อมูลก่อนใช้: ขอบที่แตะกันจะรวมเป็น component เดียว และรอยโรคที่แยกเป็นชิ้นจะถูกแบ่ง การมี instance IDs ที่เชื่อถือได้ควรปรับตัวอ่านข้อมูลแทนวิธีนี้

**Mask ว่าง:** รายงานจำนวนและเก็บใน manifest แต่ข้ามการฝึก/ประเมิน prompted lesion segmentation ไม่สร้างกรอบรอยโรคขึ้นเอง จึงไม่ได้ฝึกความสามารถปฏิเสธภาพ normal ต้องประเมิน detector และ false positives ใน pipeline แยกต่างหาก

## ติดตั้งบนเครื่อง GPU

ใช้ Python 3.12 และ PyTorch >=2.6 พร้อม torchvision รุ่นที่เข้าคู่กันและรองรับ CUDA ของเครื่อง ตาม https://pytorch.org/get-started/locally/ (Colab มักมี PyTorch อยู่แล้ว)

```bash
pip install -r sam_training/requirements.txt
```

ดาวน์โหลด checkpoint **SAM ViT-B** จาก [Meta อย่างเป็นทางการ](https://github.com/facebookresearch/segment-anything#model-checkpoints): `sam_vit_b_01ec64.pth` รุ่นนี้ไม่ใช่ YOLO `best.pt` และไม่ใช่ SAM 2/3

## 1. เตรียม manifest และแยกข้อมูล

สร้าง `groups.csv` ซึ่งมีคอลัมน์ `image,group` เช่น:

```csv
image,group
benign/benign (1).png,patient_A
benign/benign (2).png,patient_A
malignant/malignant (1).png,patient_B
```

ใช้รหัสกลุ่มแบบไม่ระบุตัวตนและรวมภาพผู้ป่วยเดียวกันไว้ด้วยกัน รวมทั้งภาพ augment ที่มาจากต้นฉบับเดียวกัน ตัวเลขในชื่อภาพไม่ได้พิสูจน์ว่าเป็นรหัสผู้ป่วย

```bash
python sam_training/data.py --data /path/to/dataset --groups groups.csv --out manifest.json
```

แบ่งประมาณ 70/15/15 ตาม **กลุ่มผู้ป่วย** ด้วย seed คงที่ ป้องกันภาพพิกเซลเหมือนกันข้ามกลุ่ม/ชุด มีการตรวจ split ซ้ำก่อนฝึก เก็บ manifest ไว้ใช้เดิม ห้ามสุ่ม test ใหม่เพื่อให้คะแนนดีขึ้น การสุ่มไม่ stratify คลาส ให้ตรวจ distribution เองและใช้ข้อมูลที่เพียงพอ หากชุดใดไม่มี positive mask จะหยุดพร้อม error

ถ้าไม่มี patient IDs ใช้ `--allow-image-split` แทน `--groups` ได้เฉพาะการทดลองเบื้องต้น โดยยังไม่สามารถอ้างความเป็นอิสระระดับผู้ป่วยได้ ข้อมูลเพียง 3 ภาพ normal ที่ส่งก่อนหน้านี้ไม่พอและจะถูกปฏิเสธ

## 2. ฝึก

```bash
python sam_training/train.py \
  --data /path/to/dataset --manifest manifest.json \
  --base /path/to/sam_vit_b_01ec64.pth \
  --out runs/sam_breast --epochs 20 --lr 0.0001 --device cuda
```

Train ใช้ box จาก annotation ขยายขอบแบบสุ่มไม่เกิน 10%; validation ใช้กรอบจริงจาก annotation ไม่สุ่ม Transform ใช้ ResizeLongestSide/preprocess/postprocess ของ SAM เพื่อให้ภาพ กล่อง และ mask ตรงกัน encoder อยู่ใน no_grad แต่ decoder ต้องมี gradient (ไม่ใช้ SamPredictor.predict เพื่อฝึก)

Outputs: `best_decoder.pt`, `history.jsonl`, `manifest.json`, `config.json` โค้ดจะไม่เขียนทับ run directory เดิม เริ่มใหม่ให้เปลี่ยน `--out` ยังไม่มี resume optimizer/epoch

ไฟล์ `best_decoder.pt` เก็บเฉพาะ decoder ต้องใช้คู่กับ base checkpoint เดิม มีการตรวจ SHA-256 ไม่สามารถแทน YOLO best.pt ใน Streamlit ได้ตรง ๆ เก็บ weights/dataset ในพื้นที่ส่วนตัว ไม่ commit ไป repository สาธารณะ

## 3. ทดสอบหลังเลือกโมเดลเสร็จ

```bash
python sam_training/train.py \
  --data /path/to/dataset --manifest runs/sam_breast/manifest.json \
  --base /path/to/sam_vit_b_01ec64.pth \
  --evaluate runs/sam_breast/best_decoder.pt --device cuda
```

รายงาน Dice/IoU แบบเฉลี่ยต่อ component ด้วย threshold logit >0 (probability >0.5) บน **test ที่ใช้กรอบจาก annotation** คะแนนนี้ประเมิน segmentation เมื่อให้ตำแหน่งถูกต้อง ไม่ใช่ผล YOLO→SAM ทั้งระบบ ไม่รวม lesion ที่ YOLO พลาดหรือ false positives และไม่ใช่ความแม่นยำการวินิจฉัย ต้องประเมิน pipeline กับภาพและกลุ่มผู้ป่วยที่ไม่เคยใช้ฝึกทั้ง YOLO และ SAM เพิ่มเติม

## 4. สร้าง mask จากภาพใหม่

ใช้กรอบ xyxy ในพิกเซลของภาพต้นฉบับ (เปลี่ยนตัวเลขให้ตรงกับภาพ):

```bash
python sam_training/predict.py --image image.png \
  --base sam_vit_b_01ec64.pth --decoder runs/sam_breast/best_decoder.pt \
  --box 50 60 180 200 --out prediction_01
```

หรือใช้ BB จาก YOLO เดิม (ติดตั้ง `ultralytics` เพิ่มใน environment ฝึก):

```bash
pip install ultralytics==8.4.150
python sam_training/predict.py --image image.png \
  --base sam_vit_b_01ec64.pth --decoder runs/sam_breast/best_decoder.pt \
  --yolo yolo_chatbot/best.pt --conf 0.5 --out prediction_02
```

ได้ `mask_001.png` ต่อกรอบ, `overlay.png`, `predictions.json` ไม่มีกรอบ=ไม่มี mask ไม่ได้ยืนยันว่าไม่มีโรค สีส้มคือคลาส malignant ที่ YOLO ทำนาย ไม่ใช่การยืนยันมะเร็งโดย SAM ไม่ใช้คะแนน IoU head ของ SAM เป็น confidence เพราะสูตรฝึกนี้ไม่ได้ calibrate head นั้น

## ตรวจโค้ด

```bash
python -m unittest discover -s sam_training -p 'test*.py'
```

ทดสอบ pairing หลาย mask, binary validation, grouped split/leakage, gradients ของ decoder, output alignment กับภาพอัตราส่วนไม่จัตุรัส, train/val loop และปฏิเสธ annotation ที่เปลี่ยนหลังสร้าง manifest

อ้างอิง API และ checkpoint: [Meta SAM](https://github.com/facebookresearch/segment-anything) โค้ดชุดนี้เป็น training recipe ที่เพิ่มให้โปรเจกต์ ไม่ใช่สูตรฝึกทางคลินิกที่ Meta รับรอง
