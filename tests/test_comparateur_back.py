import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from comparateur_back import (
    calculer_metriques,
    construire_mapping_gold,
    charger_tous_les_golds,
    evaluer_modeles_globaux,
    generer_conclusion_texte,
    generer_rapport_pdf
)

# ─────────────────────────────────────────
# 1. Tests des fonctions de calcul pur
# ─────────────────────────────────────────
def test_calculer_metriques():
    """Test le calcul des scores MAE, RMSE, R2 et du score agrégé."""
    y_true = np.array([37.0, 37.5, 38.0, 37.2])
    y_pred = np.array([37.1, 37.4, 38.1, 37.2])
    
    resultats = calculer_metriques(y_true, y_pred)
    
    assert 'mae' in resultats
    assert 'score_agg' in resultats
    assert resultats['mae'] > 0
    assert resultats['score_agg'] > 0 # Le score agrégé doit être positif

def test_generer_conclusion_texte():
    """Test la logique de décision entre l'ancien et le nouveau modèle."""
    score_A = {'score_agg': 80.0}
    
    # Égalité (différence <= 1)
    assert "similaires" in generer_conclusion_texte(score_A, {'score_agg': 80.5})
    # B meilleur (Nouveau)
    assert "NOUVEL" in generer_conclusion_texte(score_A, {'score_agg': 85.0})
    # A meilleur (Ancien)
    assert "ANCIEN" in generer_conclusion_texte(score_A, {'score_agg': 75.0})

# ─────────────────────────────────────────
# 2. Tests des interactions avec les fichiers (tmp_path)
# ─────────────────────────────────────────
def test_construire_mapping_gold(tmp_path):
    """Vérifie la création du dictionnaire liant patientId et fichier excel Gold."""
    # Création d'un faux fichier data1.csv
    df_csv = pd.DataFrame({'patientId': [101], 'autre_colonne': ['A']})
    df_csv.to_csv(tmp_path / "data1.csv", index=False)
    
    mapping = construire_mapping_gold(dossier_data=str(tmp_path))
    
    assert 101 in mapping
    assert mapping[101] == "gold_data1.xlsx"

def test_charger_tous_les_golds_succes(tmp_path):
    """Vérifie l'extraction complexe des données depuis l'Excel Gold."""
    # Création d'un faux fichier Excel avec la structure attendue par ton code
    df_excel = pd.DataFrame({
        0: ['ligne1', 'ligne2', 'ligne3'],
        1: ['Autre', 'Temp_Sonde_1', 'Ignorer'], # C'est la colonne 1 qui déclenche le filtre "Temp"
        2: [0, 37.5, 0],                         # Colonne 2 : Valeur de température
        3: ['', '2023-01-01 10:00:00', '']       # Colonne 3 : Timestamp
    })
    chemin_excel = tmp_path / "gold_data1.xlsx"
    df_excel.to_excel(chemin_excel, index=False, header=False)
    
    mapping = {101: "gold_data1.xlsx"}
    df_gold = charger_tous_les_golds(dossier_data=str(tmp_path), mapping=mapping)
    
    assert not df_gold.empty
    assert len(df_gold) == 1 # Seule la ligne contenant "Temp" doit être gardée
    assert df_gold.iloc[0]['patientId'] == 101
    assert df_gold.iloc[0]['temp_sonde'] == 37.5

def test_charger_tous_les_golds_vide(tmp_path):
    """Vérifie qu'une erreur est levée s'il n'y a aucun fichier Gold."""
    with pytest.raises(ValueError, match="Aucun fichier Gold valide trouvé"):
        charger_tous_les_golds(str(tmp_path), mapping={101: "inexistant.xlsx"})

# ─────────────────────────────────────────
# 3. Tests de l'évaluation globale (Mocking)
# ─────────────────────────────────────────
@patch('comparateur_back.charger_tous_les_golds')
@patch('comparateur_back.construire_mapping_gold')
def test_evaluer_modeles_globaux(mock_mapping, mock_charger_golds):
    """Vérifie la fusion des prédictions avec les données Gold sur la minute exacte."""
    # 1. On mock les fonctions de chargement pour renvoyer des DF maîtrisés
    mock_mapping.return_value = {}
    
    # Faux DataFrame Gold (Déjà flooré à la minute par sécurité)
    df_gold_mock = pd.DataFrame({
        'patientId': [1, 1],
        'timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:05:00'],
        'temp_sonde': [37.5, 37.8]
    })
    mock_charger_golds.return_value = df_gold_mock
    
    # Faux DataFrame de prédictions (Avec des secondes pour vérifier le floor('min'))
    df_predictions = pd.DataFrame({
        'patientId': [1, 1],
        'timestamp': ['2023-01-01 10:00:45', '2023-01-01 10:05:12'], 
        'Pred_A': [37.4, 37.9],
        'Pred_B': [37.6, 37.7]
    })
    
    # 2. Exécution
    df_merge, scores_A, scores_B = evaluer_modeles_globaux(df_predictions, dossier_data="fake_dir")
    
    # 3. Vérifications
    assert len(df_merge) == 2 # Les deux minutes doivent avoir matché
    assert 'mae' in scores_A
    assert 'score_agg' in scores_B

# ─────────────────────────────────────────
# 4. Test du PDF
# ─────────────────────────────────────────
def test_generer_rapport_pdf():
    """Vérifie que le PDF est bien généré sans crasher et retourne des bytes."""
    faux_scores = {'mae': 0.1, 'rmse': 0.2, 'r2': 0.95, 'score_agg': 98.0}
    
    resultat_pdf = generer_rapport_pdf(
        nom_comparaison="Test Unitaire",
        nom_utilisateur="Gael",
        texte_filtres="Aucun",
        volume_lignes=1000,
        nom_fichier_A="modeleA.pkl",
        nom_fichier_B="modeleB.pkl",
        scores_A=faux_scores,
        scores_B=faux_scores,
        conclusion="Conclusion de test"
    )
    
    # On s'assure que la fonction renvoie bien une chaîne de bytes (format PDF)
    assert isinstance(resultat_pdf, bytes)
    assert len(resultat_pdf) > 1000 # Un PDF vide fait au moins quelques centaines d'octets
    # On vérifie la signature d'un fichier PDF (%PDF)
    assert resultat_pdf.startswith(b'%PDF')