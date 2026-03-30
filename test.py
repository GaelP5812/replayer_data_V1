import pandas as pd
import os

DOSSIER_ENTREE = "dirty_data/"
DOSSIER_SORTIE = "data_final/"

CAPTEURS_TEMP = [
    "temperatureSensor1", "temperatureSensor2", "temperatureSensor3",
    "temperatureSensor4", "temperatureSensor5", "temperatureSensor6",
    "temperatureSensor7", "temperatureArmpitSensor", "temperatureAirSensor"
]

def traiter_patient(chemin_data, chemin_gold):
    # ── Chargement ──
    df_data = pd.read_csv(chemin_data, sep=None, engine='python')
    df_gold = pd.read_excel(chemin_gold, header=None)

    # ── Resample brassard à la minute (médiane) ──
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], errors='coerce')
    df_data = df_data.dropna(subset=['timestamp'])

    # Sauvegarder les colonnes non numériques
    cols_non_num = ['sex', 'patientId', 'armbandId', 'monitoringTabletIdentifier', 'structureId']
    cols_non_num = [c for c in cols_non_num if c in df_data.columns]
    df_non_num = df_data[cols_non_num].iloc[0]  # une seule valeur par patient

    df_data = df_data.set_index('timestamp')
    df_resampled = df_data.resample('1min').median(numeric_only=True).reset_index()
    df_resampled['timestamp'] = df_resampled['timestamp'].dt.floor('min')

    # Réinjection
    for col in cols_non_num:
        df_resampled[col] = df_non_num[col]

    # ── Extraction gold : uniquement lignes Temp ──
    df_gold.columns = range(len(df_gold.columns))
    df_temp = df_gold[df_gold[1].astype(str).str.strip() == 'Temp'].copy()
    df_temp = df_temp[[3, 2]].copy()
    df_temp.columns = ['timestamp_sonde', 'TempSonde']
    df_temp['timestamp_sonde'] = pd.to_datetime(
        df_temp['timestamp_sonde'],
        format='%d/%m/%Y %H:%M',
        errors='coerce'
    )
    df_temp = df_temp.dropna(subset=['timestamp_sonde'])
    df_temp['TempSonde'] = pd.to_numeric(df_temp['TempSonde'], errors='coerce')
    df_temp = df_temp.dropna(subset=['TempSonde'])
    df_temp['timestamp_sonde'] = df_temp['timestamp_sonde'].dt.floor('min')

    # ── Merge sur la minute ──
    df_merge = pd.merge(
        df_resampled,
        df_temp,
        left_on='timestamp',
        right_on='timestamp_sonde',
        how='inner'
    )
    df_merge = df_merge.drop(columns=['timestamp_sonde'])
    df_merge = df_merge.dropna(subset=['TempSonde'])

    # ── Nettoyage ──
    n_avant = len(df_merge)

    # scdState == 3
    df_merge = df_merge[df_merge['scdState'] == 3]
    n_scd = n_avant - len(df_merge)

    # ppgHeartRateCI > 74
    df_merge = df_merge[df_merge['ppgHeartRateCI'] > 74]
    n_ppg = len(df_merge) + n_scd - n_avant + (n_avant - n_scd - len(df_merge))
    n_ppg = n_avant - n_scd - len(df_merge) - 0
    
    # capteurs 1 et 7 pas NaN
    df_avant = df_merge.copy()
    df_merge = df_merge[df_merge['temperatureSensor1'].notna() & df_merge['temperatureSensor7'].notna()]
    n_nan = len(df_avant) - len(df_merge)

    # variation > 0.5°C sur les 9 capteurs
    masque_variation = pd.Series(False, index=df_merge.index)
    for capteur in CAPTEURS_TEMP:
        if capteur in df_merge.columns:
            masque_variation |= df_merge[capteur].diff().abs() > 0.5
    df_avant = df_merge.copy()
    df_merge = df_merge[~masque_variation]
    n_variation = len(df_avant) - len(df_merge)

    print(f"  {len(df_merge)} lignes finales (scd:-{n_scd} ppg:-{n_ppg} nan:-{n_nan} var:-{n_variation})")

    return df_merge


def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    for i in range(1, 30):
        chemin_data = os.path.join(DOSSIER_ENTREE, f"data{i}.csv")
        chemin_gold = os.path.join(DOSSIER_ENTREE, f"gold_data{i}.xlsx")

        if not os.path.exists(chemin_data):
            print(f"data{i}.csv : introuvable, ignoré")
            continue
        if not os.path.exists(chemin_gold):
            print(f"gold_data{i}.xlsx : introuvable, ignoré")
            continue

        print(f"Traitement data{i}...")
        try:
            df_result = traiter_patient(chemin_data, chemin_gold)
            chemin_sortie = os.path.join(DOSSIER_SORTIE, f"data{i}.csv")
            df_result.to_csv(chemin_sortie, index=False)
        except Exception as e:
            print(f"  Erreur : {e}")

if __name__ == "__main__":
    main()
