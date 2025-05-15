import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Excel Visualizer", layout="wide")
st.title("📊 Excel-Daten visualisieren")

# Excel-Datei Upload
uploaded_file = st.file_uploader("📂 Excel-Datei hochladen", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ Datei erfolgreich geladen.")

        # Erste Vorschau
        st.subheader("🔍 Datenvorschau")
        st.dataframe(df.head())

        # Auswahl einer Spalte für Visualisierung
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox("📈 Wähle eine numerische Spalte für die Visualisierung", numeric_columns)

            # Diagramm anzeigen
            st.subheader(f"📊 Histogramm für: {selected_column}")
            fig, ax = plt.subplots()
            ax.hist(df[selected_column].dropna(), bins=20, color="skyblue", edgecolor="black")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Häufigkeit")
            ax.set_title(f"Verteilung von {selected_column}")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Keine numerischen Spalten gefunden.")

    except Exception as e:
        st.error(f"❌ Fehler beim Einlesen der Datei: {e}")
else:
    st.info("Bitte lade eine Excel-Datei hoch, um fortzufahren.")
