"""Image input, inference output, mask measurements and downloadable reports."""
import hashlib
import io
import math
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageOps
from chat import NOTICE, summary


def read_image(raw):
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("ไฟล์ใหญ่เกิน 10 MB")
    with Image.open(io.BytesIO(raw)) as source:
        if source.format not in {"PNG", "JPEG"}:
            raise ValueError("รองรับเฉพาะภาพ PNG และ JPEG")
        if source.width * source.height > 20_000_000:
            raise ValueError("ภาพต้องมีไม่เกิน 20 ล้านพิกเซล")
        picture = ImageOps.exif_transpose(source).convert("RGB")
        if all(low == high for low, high in picture.getextrema()):
            raise ValueError("ภาพมีสีเดียวทั้งหมด กรุณาเลือกภาพที่มีข้อมูลภาพ")
    return picture


def polygon_features(points):
    if len(points) < 3:
        return None
    pairs = list(zip(points, points[1:] + points[:1]))
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in pairs)) / 2
    perimeter = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in pairs)
    if area <= 0 or perimeter <= 0:
        return None
    return {"area_px2": round(area, 2), "perimeter_px": round(perimeter, 2),
            "circularity": round(4 * math.pi * area / perimeter ** 2, 4)}


def prediction_data(prediction, image_hash, threshold, model_hash, task):
    polygons = prediction.masks.xy if prediction.masks is not None else []
    detections = []
    for i, box in enumerate(prediction.boxes):
        item = {"id": i + 1, "label": prediction.names[int(box.cls.item())],
                "confidence": float(box.conf.item()), "bbox": box.xyxy[0].cpu().tolist()}
        if i < len(polygons):
            points = polygons[i].tolist()
            item["polygon"] = points
            item["shape"] = polygon_features(points)
        detections.append(item)
    return {"image_sha256": image_hash, "model_sha256": model_hash, "task": task,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "image_height": prediction.orig_shape[0], "image_width": prediction.orig_shape[1],
            "threshold": threshold, "detections": detections}


def render_overlay(picture, detections, show_boxes=True, show_masks=True, selected=None):
    base = picture.convert("RGBA")
    overlay = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(overlay)
    for i, d in enumerate(detections, 1):
        if selected is not None and i != selected:
            continue
        color = (67, 228, 206) if d["label"] == "benign" else (255, 193, 101)
        points = d.get("polygon", [])
        if show_masks and len(points) >= 3:
            draw.polygon([tuple(p) for p in points], fill=(*color, 60), outline=(*color, 255), width=2)
        if show_boxes:
            x1, y1, x2, y2 = d["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=3)
            text = f"{i}: {d['label']} {d['confidence']:.3f}"
            tx, ty = max(0, x1), max(0, y1 - 18)
            bounds = draw.textbbox((tx, ty), text)
            draw.rectangle(bounds, fill=(15, 25, 40, 230))
            draw.text((tx, ty), text, fill=(*color, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def image_png(picture):
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def make_report(data):
    lines = ["# รายงานผลโมเดลเพื่อการศึกษา", "", NOTICE, "",
             "เวลาประมวลผล (UTC): " + data["created_at_utc"],
             "Image SHA-256: " + data["image_sha256"],
             "Model SHA-256: " + data["model_sha256"],
             f"เกณฑ์กรอง confidence: {data['threshold']:.2f}", "", summary(data), "",
             "## รายละเอียดกรอบ"]
    for i, d in enumerate(data["detections"], 1):
        lines.append(f"- กรอบ {i}: bbox (x1,y1,x2,y2) = " + ", ".join(f"{v:.1f}" for v in d["bbox"]) + " px")
        if d.get("shape"):
            s = d["shape"]
            lines.append(f"  พื้นที่จาก contour {s['area_px2']} px²; ความยาวขอบ {s['perimeter_px']} px; circularity {s['circularity']}")
    lines += ["", "ค่าขนาดอยู่ในพิกเซล ไม่มีการแปลงเป็นมิลลิเมตรหรือคะแนน BI-RADS",
              "โมเดล detection ไม่ให้ mask; รายงานนี้ไม่ใช้ BB คำนวณพื้นที่รอยโรค",
              "ยังไม่มีผลประเมินกับชุดทดสอบผู้ป่วยจริงในรุ่นนี้"]
    return "\n".join(lines)
