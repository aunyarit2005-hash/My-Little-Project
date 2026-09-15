"""UI and real-checkpoint smoke tests using synthetic images, not clinical validation."""
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_initial_page_without_secrets(self):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "BreastVision Lab")

    def test_inference_chat_downloads_and_invalidation(self):
        raw = io.BytesIO()
        Image.fromarray(np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(raw, format="PNG")
        with patch("streamlit.file_uploader", return_value=raw):
            app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=30)
            app.checkbox(key="confirmed").check().run()
            next(b for b in app.button if b.label == "ประมวลผลภาพ").click().run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            result = app.session_state["result"]
            self.assertEqual(result["task"], "detect")
            self.assertEqual(len(result["model_sha256"]), 64)
            self.assertEqual(len(app.get("download_button")), 3)
            # Deterministic UI fixture after exercising the real inference path.
            result["detections"] = [
                {"label": "benign", "confidence": .87, "bbox": [1, 2, 20, 30]},
                {"label": "malignant", "confidence": .63, "bbox": [40, 50, 60, 70]},
            ]
            app.session_state["result"] = result
            app.run()
            app.selectbox(key="selected_frame").select(2).run()
            app.button(key="quick_ดูคะแนน").click().run()
            reply = app.session_state["messages"][-1]["content"]
            self.assertIn("0.630", reply)
            self.assertNotIn("0.870", reply)
            self.assertEqual(len(app.exception), 0)
            app.slider[0].set_value(.65).run()
            self.assertIsNone(app.session_state.filtered_state.get("result"))
            self.assertIsNone(app.session_state.filtered_state.get("messages"))
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
