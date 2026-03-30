import pandas as pd
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import streamlit as st
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fonction de scoring
def calculer_metriques(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    score_r2   = max(0, r2 * 27)
    score_rmse = max(0, (1 - rmse / 1.5) * 45)
    score_mae  = max(0, (1 - mae  / 1) * 28)
    score_agg  = score_r2 + score_rmse + score_mae

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'score_agg': score_agg}


def evaluer_modeles_globaux(df_predictions):
    df_pred = df_predictions.copy()
    df_base = st.session_state["df_filtre"]
    
    # Récupérer TempSonde depuis df_filtre et merger sur patientId + timestamp
    df_gold = df_base[['timestamp', 'patientId', 'TempSonde']].copy()
    df_gold['TempSonde'] = pd.to_numeric(df_gold['TempSonde'], errors='coerce')
    df_gold['timestamp'] = pd.to_datetime(df_gold['timestamp'], errors='coerce')
    df_gold = df_gold.dropna(subset=['timestamp', 'TempSonde'])
    df_gold = df_gold.rename(columns={'TempSonde': 'temp_sonde'})

    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'], errors='coerce')
    df_pred = df_pred.dropna(subset=['timestamp'])

    df_merge = pd.merge(
        df_pred,
        df_gold,
        on=['patientId', 'timestamp'],
        how='inner'
    )

    df_merge = df_merge.dropna(subset=['temp_sonde', 'Pred_A', 'Pred_B'])

    if df_merge.empty:
        raise ValueError("Échec : Aucune correspondance entre les prédictions et le gold standard.")

    scores_A = calculer_metriques(df_merge['temp_sonde'], df_merge['Pred_A'])
    scores_B = calculer_metriques(df_merge['temp_sonde'], df_merge['Pred_B'])

    return df_merge, scores_A, scores_B

def generer_conclusion_texte(scores_A, scores_B):
    diff = scores_B['score_agg'] - scores_A['score_agg']
    if abs(diff) <= 1.0:
        return f"Les deux algorithmes ont des performances similaires."
    elif diff > 0:
        return f"Le NOUVEL algorithme est le plus performant."
    else:
        return f"L'ANCIEN algorithme reste le plus performant."

def generer_rapport_pdf(nom_comparaison, nom_utilisateur, texte_filtres, volume_lignes, nom_fichier_A, nom_fichier_B, scores_A, scores_B, conclusion):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. On remplace "Arial" par "Helvetica" (qui est native)
    pdf.set_font("Helvetica", "B", 18)
    
    # 2. On remplace ln=True par new_x=XPos.LMARGIN, new_y=YPos.NEXT
    pdf.cell(0, 15, "Rapport de Comparaison d'Algorithmes", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Informations generales", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Nom du projet : {nom_comparaison}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Utilisateur : {nom_utilisateur}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Date de l'analyse : {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Filtres appliques : {texte_filtres}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Volume traite : {volume_lignes} lignes", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(4)
    pdf.cell(0, 6, f"Fichier Modele A (Ancien) : {nom_fichier_A}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Fichier Modele B (Nouveau) : {nom_fichier_B}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Comparaison des performances", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    for label, scores in [("ANCIEN ALGORITHME", scores_A), ("NOUVEL ALGORITHME", scores_B)]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f" {label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 6, f"   - Erreur Absolue Moyenne (MAE) : {scores['mae']:.4f} ", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 6, f"   - Erreur Quadratique (RMSE)    : {scores['rmse']:.4f} ", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 6, f"   - Coefficient de Correlation (R2) : {scores['r2']:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, f"   => SCORE AGREGE : {scores['score_agg']:.2f} / 100", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Conclusion de l'expertise", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 7, conclusion)

    pdf_output = pdf.output() 
    
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    elif isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    
    return pdf_output