import pandas as pd
import joblib
import xgboost as xgb
from model import feature_engineering

# ─────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────
COLONNES_CAPTEURS = [
    "temperatureSensor1", "temperatureSensor2", "temperatureSensor3",
    "temperatureSensor4", "temperatureSensor5", "temperatureSensor6",
    "temperatureSensor7", "temperatureArmpitSensor", "temperatureAirSensor"
]

COLS_SMOOTH = [
    'Temp_peau_max', 'temperatureAirSensor',
    'temperatureArmpitSensor', 'ppgHeartRate',
    'heartRateVariability'
]
COLS_DIFF = [
    ("Temp_peau_max_liss", "temperatureArmpitSensor_liss"),
    ("temperatureAirSensor_liss", "temperatureArmpitSensor_liss")
]

# ─────────────────────────────────────────
# Fonctions de préparation
# ─────────────────────────────────────────
def preparer_df(df_base):
    if df_base.empty:
        raise ValueError("Le DataFrame est vide avant préparation.")
    
    # Nettoyage du timestamp et des colonnes vides
    df_base['timestamp'] = pd.to_datetime(df_base['timestamp'], errors='coerce')
    df_base = df_base.dropna(subset=['timestamp'])
    df_base = df_base.dropna(axis=1, how='all')

    dfs_prets = []

    for patient_id, df_patient in df_base.groupby('patientId'):
        
        # Gestion du sexe
        valeur_sexe = df_patient['sex'].iloc[0] if 'sex' in df_patient.columns else 'H'
        valeur_sexe = 'H' if valeur_sexe == 'M' else valeur_sexe
        df_patient = df_patient.copy()
        df_patient['sex'] = valeur_sexe

        # Feature Engineering
        try:
            df_feat = feature_engineering(
                input_data=df_patient,
                freq='1min',
                smoothing_period="5min",
                cols_smooth=COLS_SMOOTH,
                cols_diff=COLS_DIFF,
                mode='complete'
            )
            if df_feat is not None and not df_feat.empty:
                dfs_prets.append(df_feat)
        except Exception:
            continue

    if not dfs_prets:
        raise ValueError("Aucun dataset n'a pu être traité. Vérifiez le format des données.")
    
    df_final = pd.concat(dfs_prets, ignore_index=True)
    
    cols_a_virer = [c for c in df_final.columns if df_final[c].isna().all()]
    if cols_a_virer:
        df_final = df_final.drop(columns=cols_a_virer)

    return df_final

# ─────────────────────────────────────────
# Inférence Double
# ─────────────────────────────────────────
def executer_double_inference(df_base, fichier_A, fichier_B):
    df_pret = preparer_df(df_base.copy())
    if len(df_pret) == 0:
        raise ValueError("Le DataFrame est vide après le Feature Engineering.")

    try:
        modele_A = joblib.load(fichier_A)
    except BaseException:
        raise ValueError(f"Fichier illisible : '{fichier_A.name}'.")

    try:
        modele_B = joblib.load(fichier_B)
    except BaseException:
        raise ValueError(f"Fichier illisible : '{fichier_B.name}'.")

    # Prédiction Modèle A
    try:
        cols_A = modele_A.feature_names
        if all(col in df_pret.columns for col in cols_A):
            df_A_propre = df_pret.dropna(subset=cols_A)
            if not df_A_propre.empty:
                pred_A = modele_A.predict(xgb.DMatrix(df_A_propre[cols_A]))
                df_pret.loc[df_A_propre.index, 'Pred_A'] = pred_A
            else:
                df_pret['Pred_A'] = None
        else:
            raise ValueError(f"Colonnes manquantes pour '{fichier_A.name}'.")
    except ValueError:
        raise
    except Exception:
        raise ValueError(f"Algorithme incompatible : '{fichier_A.name}'.")

    # Prédiction Modèle B
    try:
        cols_B = modele_B.feature_names
        if all(col in df_pret.columns for col in cols_B):
            df_B_propre = df_pret.dropna(subset=cols_B)
            if not df_B_propre.empty:
                pred_B = modele_B.predict(xgb.DMatrix(df_B_propre[cols_B]))
                df_pret.loc[df_B_propre.index, 'Pred_B'] = pred_B
            else:
                df_pret['Pred_B'] = None
        else:
            raise ValueError(f"Colonnes manquantes pour '{fichier_B.name}'.")
    except ValueError:
        raise
    except Exception:
        raise ValueError(f"Algorithme incompatible : '{fichier_B.name}'.")

    return df_pret.dropna(subset=['Pred_A', 'Pred_B'], how='all')