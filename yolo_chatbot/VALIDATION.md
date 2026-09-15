# Validation — 2026-09-15

## Current Cloud-ready revision

- Python 3.12; Streamlit 1.63.0, Ultralytics 8.4.150, Torch 2.14.0+cpu, torchvision 0.29.0+cpu.
- 24 tests passed: original 8 chat tests, 14 image/chat/LLM configuration tests, and 2 Streamlit AppTest checks.
- Real `best.pt` loaded on CPU. AppTest exercised synthetic PNG input, confirmation, actual inference, PNG/JSON/Markdown download controls, selected-frame chat and threshold invalidation without application exceptions.
- Synthetic detections verified that a selected frame returns its actual score, explicit frame numbers override selection, unavailable frames are not invented, and detection-only results do not fabricate masks.
- Hosted LLM requests were tested with mocked responses, including timeout and invalid citation fallback. No live API key or running Ollama service was supplied; live LLM generation is not verified. LLM is disabled by default.
- Polygon measurements verified on known geometry. No trained segmentation model or real ultrasound validation set was supplied. Real segmentation performance and clinical accuracy are not established.
- Browser visual QA and final hosted deployment are recorded separately when completed; AppTest does not prove the Cloud deployment is live.

## Previous validation record (2026-09-14, before RAG update)

- Python syntax checks passed.
- Eight deterministic chatbot tests passed (empty result, mixed labels, benign limitations, confidence, treatment boundary, counts, follow-up, unknown question).
- Uploaded best.pt loaded with Ultralytics 8.4.150 / Torch 2.14.0 on CPU. Classes verified: benign, malignant; task detect.
- Synthetic image inference and plotting succeeded. A blank black image produced one detection at the library default threshold: this is an out-of-distribution false detection, NOT a clinical result. The app now rejects uniform-color images; this is not a general quality or modality detector.
- Streamlit 1.63.0 AppTest: initial UI, injected synthetic PNG input, confirmation, real model inference, chat reply, and threshold-change invalidation passed without application exceptions. File chooser was replaced by an in-memory input adapter for this test.
- No real ultrasound validation set was supplied. No clinical accuracy claim is made.
- Ollama integration is implemented but not live-tested; no local language model was available. Default chat runs without Ollama.
- Browser visual QA and Windows execution have not been performed.
