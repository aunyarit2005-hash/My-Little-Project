"""Prototype-inspired four-panel presentation; native Streamlit controls."""
import streamlit as st


def apply_ui():
    st.markdown("""
    <style>
    .stApp { background: #151e23; color: #e9f0f4; }
    [data-testid="stHeader"] { background: #151e23; }
    [data-testid="stSidebar"] { background: #202c34; }
    .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.9rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.2rem !important; padding-top: 0 !important; }
    [data-testid="stCaptionContainer"] { color: #a8c6d6; line-height: 1.65; }
    .st-key-image_panel, .st-key-features_panel,
    .st-key-chat_panel, .st-key-sources_panel {
        background: #202c34; border: 1px solid #3a515d !important;
        border-radius: 20px !important; padding: 24px !important;
    }
    .st-key-image_panel, .st-key-features_panel { min-height: 590px; }
    .st-key-chat_panel, .st-key-sources_panel { min-height: 480px; }
    .st-key-image_panel [data-testid="stImage"] { background: #111d23; width: 100%; }
    .st-key-image_panel [data-testid="stImage"] img { height: 355px; object-fit: contain; width: 100%; }
    .image-placeholder { height: 390px; display: flex; flex-direction: column;
        align-items: center; justify-content: center; background: #111d23;
        color: #d8e8ef; border-radius: 8px; font-size: 1.25rem; }
    .image-placeholder span { font-size: .9rem; color: #a8c6d6; margin-top: 14px; }
    hr { border-color: #3a515d !important; margin: .6rem 0 !important; }
    [data-testid="stMetric"] { padding: 8px 0; }
    [data-testid="stMetricLabel"] { color: #a8c6d6; }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #eef5f8; }
    [data-testid="stChatMessage"] { background: #1d4142; border-radius: 16px; padding: 16px; }
    [data-testid="stChatMessage"] p { line-height: 1.8; overflow-wrap: anywhere; }
    [data-testid="stChatInput"] { border-color: #3a515d; }
    .stButton > button, .stDownloadButton > button { border: 1px solid #3a515d;
        background: #202c34; border-radius: 13px; min-height: 46px; }
    .stButton > button[kind="primary"] { background: #76d9cf; color: #10282d; }
    a { color: #79ded8 !important; }
    @media(max-width: 768px) {
        .block-container { padding: 1rem; }
        .st-key-image_panel, .st-key-features_panel,
        .st-key-chat_panel, .st-key-sources_panel { min-height: auto; padding: 16px !important; }
        .st-key-image_panel [data-testid="stImage"] img { height: 290px; }
    }
    </style>
    """, unsafe_allow_html=True)
