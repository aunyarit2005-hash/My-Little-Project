import io
import json
import unittest
from unittest.mock import patch

from PIL import Image
from chat import detection_answer
from image_analysis import polygon_features, read_image, render_overlay, image_png
from llm import LLMConfig, load_config, complete_json
from rag import rag_answer, generate, retrieve

DATA = {"detections": [
    {"label": "benign", "confidence": 0.87, "bbox": [1, 2, 20, 30]},
    {"label": "malignant", "confidence": 0.63, "bbox": [40, 50, 60, 70]},
]}


class CloudTests(unittest.TestCase):
    def test_question_returns_actual_scores(self):
        text = rag_answer("คะแนนกรอบที่ตรวจพบเท่าไร", DATA)
        self.assertIn("0.870", text)
        self.assertIn("0.630", text)
        self.assertIn("ไม่ใช่เปอร์เซ็นต์โอกาส", text)

    def test_explicit_frame_overrides_selection(self):
        text = rag_answer("คะแนนกรอบที่ ๑ เท่าไร", DATA, selected=2)
        self.assertIn("0.870", text)
        self.assertNotIn("0.630", text)

    def test_selected_frame_and_missing_frame(self):
        self.assertIn("0.630", detection_answer("คะแนนเท่าไร", DATA, selected=2))
        self.assertIn("ไม่พบกรอบที่ 9", rag_answer("กรอบที่ 9", DATA))

    def test_empty_does_not_imply_normal(self):
        self.assertIn("ไม่ได้หมายความว่าภาพปกติ", rag_answer("คะแนนเท่าไร", {"detections": []}))

    def test_detection_has_no_fake_mask(self):
        self.assertIn("ไม่มี mask", rag_answer("อธิบายรูปร่าง mask", DATA))

    def test_clinical_boundary_never_calls_llm(self):
        with patch("rag.complete_json") as call:
            text = rag_answer("เป็นมะเร็งไหม", DATA, LLMConfig("openai", "test", "https://example.com", "secret"))
        call.assert_not_called()
        self.assertIn("ไม่สามารถยืนยันโรค", text)

    def test_default_never_calls_llm(self):
        with patch("rag.complete_json") as call:
            text = rag_answer("อัลตราซาวด์คืออะไร", DATA)
        call.assert_not_called()
        self.assertIn("US-ROLE", text)

    def test_llm_failure_preserves_actual_score_and_evidence(self):
        with patch("rag.complete_json", side_effect=TimeoutError):
            text = rag_answer("คะแนนกรอบที่ 1", DATA, LLMConfig("openai", "test", "https://example.com", "secret"))
        self.assertIn("0.870", text)
        self.assertIn("LLM ไม่พร้อม", text)
        self.assertIn("GUO2017", text)

    def test_bad_citation_falls_back(self):
        with patch("rag.complete_json", return_value={"paragraphs": [{"text": "example", "source_ids": ["INVENTED"]}]}):
            text = rag_answer("confidence คืออะไร", DATA, LLMConfig("openai", "test", "https://example.com", "secret"))
        self.assertNotIn("INVENTED", text)
        self.assertIn("LLM ไม่พร้อม", text)

    def test_llm_receives_only_question_and_evidence(self):
        with patch("rag.complete_json", return_value={"paragraphs": [{"text": "คำอธิบาย", "source_ids": ["GUO2017"]}]}) as call:
            generate("confidence", retrieve("confidence"), LLMConfig("openai", "test"))
        payload = json.loads(call.call_args.args[0][1]["content"])
        self.assertEqual(set(payload), {"QUESTION", "EVIDENCE"})

    def test_configuration_rejects_missing_key_and_remote_http(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(load_config().enabled)
            with self.assertRaises(ValueError):
                load_config({"provider": "openai", "model": "test"})
            with self.assertRaises(ValueError):
                load_config({"provider": "ollama", "model": "test", "base_url": "http://remote.invalid"})
            self.assertTrue(load_config({"provider": "ollama", "model": "test"}).enabled)
        self.assertNotIn("private-key", repr(LLMConfig(api_key="private-key")))

    def test_hosted_request_uses_configured_endpoint(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": '{"paragraphs": []}'}}]}).encode()
        with patch("llm.build_opener") as opener:
            opener.return_value.open.return_value = response
            result = complete_json([], LLMConfig("openai", "test", "https://example.com/v1", "private-key"))
        request = opener.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer private-key")
        self.assertEqual(result, {"paragraphs": []})

    def test_known_polygon_measurements(self):
        shape = polygon_features([[0, 0], [10, 0], [10, 10], [0, 10]])
        self.assertEqual(shape["area_px2"], 100)
        self.assertEqual(shape["perimeter_px"], 40)
        self.assertAlmostEqual(shape["circularity"], 0.7854)
        self.assertIsNone(polygon_features([[0, 0], [1, 1]]))

    def test_image_checks_and_overlay(self):
        with self.assertRaises(ValueError):
            read_image(image_png(Image.new("RGB", (80, 80))))
        picture = Image.new("RGB", (80, 80))
        picture.putpixel((1, 1), (255, 255, 255))
        read_image(image_png(picture))
        self.assertEqual(render_overlay(picture, DATA["detections"], False, False).tobytes(), picture.tobytes())
        self.assertNotEqual(render_overlay(picture, DATA["detections"]).tobytes(), picture.tobytes())


if __name__ == "__main__":
    unittest.main()
