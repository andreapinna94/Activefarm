# ------------------------------------------------
# IMPORT
# ------------------------------------------------
import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv                 #  <-- NEW
from openai import OpenAI


st.title("Visualizzatore CSV")

# Caricamento file CSV
uploaded_file = st.file_uploader("Carica un file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File caricato con successo!")
        st.write("Anteprima dei dati:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
else:
    st.info("Carica un file CSV per visualizzare i dati.")
