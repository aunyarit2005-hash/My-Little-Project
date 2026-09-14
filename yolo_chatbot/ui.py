
import streamlit as st


def apply_ui():
    st.markdown("""
    <style>
    .stApp {
        background: #0b1220;
        color: #e8edf7;
    }
    [data-testid="stSidebar"] {
        background: #111c30;
        border-right: 1px solid #293750;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1 {
        font-weight: 750 !important;
        letter-spacing: -0.02em;
    }
    h2, h3 { color: #e8edf7 !important; }
    [data-testid="stCaptionContainer"] {
        color: #afbdd2;
        line-height: 1.7;
    }
    [data-testid="stMetric"] {
        background: #142138;
        border: 1px solid #304462;
        border-radius: 16px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: #bbcbe1; }
    [data-testid="stMetricValue"] { color: #70d7ff; }
    [data-testid="stChatMessage"] {
        background: #142138;
        border: 1px solid #293d5c;
        border-radius: 18px;
        margin-bottom: 14px;
        padding: 18px;
    }
    [data-testid="stChatMessage"] p {
        line-height: 1.85;
        overflow-wrap: anywhere;
    }
    [data-testid="stFileUploader"] {
        border: 1px dashed #47668f;
        border-radius: 16px;
        padding: 12px;
    }
    [data-testid="stExpander"] {
        border-radius: 14px;
        border-color: #304462;
    }
    .stButton > button[kind="primary"] {
        background: #1684d9;
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 650;
    }
    a { color: #7bd9ff !important; }
    @media(max-width: 768px) {
        .block-container { padding: 1rem; }
        [data-testid="stChatMessage"] { padding: 12px; }
    }
    </style>
    """, unsafe_allow_html=True)
