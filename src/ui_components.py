import streamlit as st
from pathlib import Path

def kpi_card(title, value):
    card_html = f"""
        <div style="background-color: transparent; 
                    padding:12px 20px; 
                    border-radius:16px; 
                    border:1px solid #334155; 
                    text-align:center; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
            <p style="font-size:14px; color:#94A3B8; margin-bottom:4px; font-weight:500; margin-top:0;">{title}</p>
            <h1 style="font-size:30px; font-weight:700; color:#F8FAFC; margin:0;">{value}</h1>
        </div>
        """
    st.markdown(card_html, unsafe_allow_html=True)

def apply_custom_css():
    # Adjust path if your app.py is in the root and style.css is in src/
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"CSS file not found at: {css_path}")