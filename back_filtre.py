import pandas as pd
import os
import streamlit as st

@st.cache_data
def charger_brassards(dossier="data"):
    dfs = []
    for i in range(1, 30):
        chemin = os.path.join(dossier, f"data{i}.csv")
        if os.path.exists(chemin):
            df = pd.read_csv(chemin, sep=None, engine='python')
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)