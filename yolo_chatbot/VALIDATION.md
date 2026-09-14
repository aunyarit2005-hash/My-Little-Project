# Validation — 2026-09-14

- Python syntax checks passed.
- Eight deterministic chatbot tests passed (empty result, mixed labels, benign limitations, confidence, treatment boundary, counts, follow-up, unknown question).
- Uploaded best.pt loaded with Ultralytics 8.4.150 / Torch 2.14.0 on CPU. Classes verified: benign, malignant; task detect.
- Synthetic image inference and plotting succeeded. A blank black image produced one detection at the library default threshold: this is an out-of-distribution false detection, NOT a clinical result. The app now rejects uniform-color images; this is not a general quality or modality detector.
- Streamlit 1.63.0 AppTest: initial UI, injected synthetic PNG input, confirmation, real model inference, chat reply, and threshold-change invalidation passed without application exceptions. File chooser was replaced by an in-memory input adapter for this test.
- No real ultrasound validation set was supplied. No clinical accuracy claim is made.
- Ollama integration is implemented but not live-tested; no local language model was available. Default chat runs without Ollama.
- Browser visual QA and Windows execution have not been performed.
