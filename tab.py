
import streamlit as st

st.title("Beispiel mit Tabs")

tab1, tab2, tab3 = st.tabs(["📊 Daten", "⚙️ Einstellungen", "ℹ️ Info"])

with tab1:
    st.subheader("Datenanalyse")
    st.line_chart([1, 2, 3, 2, 4])

with tab2:
    st.subheader("Einstellungen")
    st.slider("Wähle einen Wert", 0, 100, 50)

with tab3:
    st.subheader("Information")
    st.markdown("Diese App wurde mit Streamlit gebaut.")
