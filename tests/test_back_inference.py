import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from back_inference import preparer_df, executer_double_inference

# ─────────────────────────────────────────
# Fixtures (Données de test réutilisables)
# ─────────────────────────────────────────
@pytest.fixture
def df_brut_test():
    """Génère un faux DataFrame de départ pour les tests."""
    return pd.DataFrame({
        'timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:01:30', '2023-01-01 10:05:00'],
        'patientId': [1, 1, 2],
        'sex': ['M', 'M', 'F'], # 'M' devra être transformé en 'H'
        'temperatureAirSensor': [20.0, 20.5, 21.0]
    })

@pytest.fixture
def df_prepare_test():
    """Génère un faux DataFrame simulant la sortie de preparer_df."""
    return pd.DataFrame({
        'patientId': [1, 2],
        'feature1': [10.5, 11.0],
        'feature2': [1.2, 1.5]
    })

# ─────────────────────────────────────────
# Tests pour preparer_df
# ─────────────────────────────────────────
def test_preparer_df_vide():
    """Vérifie que la fonction crashe proprement si on lui donne un DF vide."""
    with pytest.raises(ValueError, match="Le DataFrame est vide avant préparation"):
        preparer_df(pd.DataFrame())


@patch('back_inference.feature_engineering')
def test_preparer_df_succes(mock_feature_engineering, df_brut_test):
    """Vérifie le nettoyage et la préparation normale des données."""
    # On dit au mock de renvoyer simplement le DataFrame qu'il reçoit
    mock_feature_engineering.side_effect = lambda input_data, **kwargs: input_data

    df_resultat = preparer_df(df_brut_test)

    # Vérifications
    assert not df_resultat.empty
    assert 'timestamp' in df_resultat.columns # Car il est passé en index lors du resample
    # Vérifie que le sexe 'M' a bien été transformé en 'H' pour le patient 1
    assert df_resultat[df_resultat['patientId'] == 1]['sex'].iloc[0] == 'H'


@patch('back_inference.feature_engineering')
def test_preparer_df_erreur_engineering(mock_feature_engineering, df_brut_test):
    """Vérifie que si feature_engineering plante pour tous les patients, ça lève une erreur."""
    # On simule une erreur à chaque appel de feature_engineering
    mock_feature_engineering.side_effect = Exception("Erreur de calcul")

    with pytest.raises(ValueError, match="Aucun dataset n'a pu être traité"):
        preparer_df(df_brut_test)


# ─────────────────────────────────────────
# Tests pour executer_double_inference
# ─────────────────────────────────────────
@patch('back_inference.preparer_df')
@patch('back_inference.joblib.load')
@patch('back_inference.xgb.DMatrix')
def test_executer_double_inference_succes(mock_dmatrix, mock_joblib_load, mock_preparer_df, df_brut_test, df_prepare_test):
    """Test le flux complet de prédiction avec de faux modèles."""

    # 1. On force preparer_df à renvoyer notre faux DF préparé
    mock_preparer_df.return_value = df_prepare_test

    # 2. On crée de faux modèles A et B
    mock_modele_A = MagicMock()
    mock_modele_A.feature_names = ['feature1']
    mock_modele_A.predict.return_value = [0.8, 0.9] # Fausses prédictions

    mock_modele_B = MagicMock()
    mock_modele_B.feature_names = ['feature2']
    mock_modele_B.predict.return_value = [0.2, 0.3]

    # joblib.load va être appelé 2 fois (pour A puis B). On donne les 2 faux modèles dans l'ordre.
    mock_joblib_load.side_effect = [mock_modele_A, mock_modele_B]

    # Exécution de la fonction testée
    df_final = executer_double_inference(df_brut_test, "chemin_faux_A.pkl", "chemin_faux_B.pkl")

    # Vérifications
    assert 'Pred_A' in df_final.columns
    assert 'Pred_B' in df_final.columns
    assert list(df_final['Pred_A']) == [0.8, 0.9]
    assert list(df_final['Pred_B']) == [0.2, 0.3]
    assert mock_joblib_load.call_count == 2 # Vérifie qu'on a bien chargé les deux modèles


def test_executer_double_inference_empty_after_prep(monkeypatch, df_brut_test):
    """Si preparer_df renvoie un DF vide, on lève une erreur."""
    monkeypatch.setattr('back_inference.preparer_df', lambda x: pd.DataFrame())
    with pytest.raises(ValueError, match="Le DataFrame est vide après le Feature Engineering"):
        executer_double_inference(df_brut_test, "a.pkl", "b.pkl")


def test_preparer_df_removes_all_na_columns(monkeypatch, df_brut_test):
    """Vérifie qu'on supprime les colonnes entièrement NA après concat."""
    # Mock feature_engineering to return a df with an all-NA column
    def fake_fe(input_data, **kwargs):
        df = input_data.copy()
        df['allna'] = pd.NA
        return df

    monkeypatch.setattr('back_inference.feature_engineering', fake_fe)
    df_out = preparer_df(df_brut_test)
    assert 'allna' not in df_out.columns


def test_executer_double_inference_handles_missing_features(monkeypatch, df_brut_test, df_prepare_test):
    """Si un modèle attend une feature absente, ses prédictions deviennent None pour tous."""
    # preparer_df retourne df_prepare_test
    monkeypatch.setattr('back_inference.preparer_df', lambda x: df_prepare_test)

    # modele A demande feature1 (present), modele B demande feature_missing
    mock_modele_A = MagicMock(); mock_modele_A.feature_names = ['feature1']; mock_modele_A.predict.return_value = [0.5, 0.6]
    mock_modele_B = MagicMock(); mock_modele_B.feature_names = ['feature_missing']; mock_modele_B.predict.return_value = [0.1, 0.2]

    monkeypatch.setattr('back_inference.joblib', MagicMock(load=MagicMock(side_effect=[mock_modele_A, mock_modele_B])))
    monkeypatch.setattr('back_inference.xgb', MagicMock(DMatrix=lambda df: df))

    df_final = executer_double_inference(df_brut_test, 'a', 'b')
    assert 'Pred_A' in df_final.columns
    assert 'Pred_B' in df_final.columns
    # Since modele_B's feature is missing, Pred_B should be all None
    assert df_final['Pred_B'].isnull().all()


def test_preparer_df_sex_default_when_missing(monkeypatch):
    """Si la colonne sex est absente, on utilise 'H' par défaut."""
    df = pd.DataFrame({
        'timestamp': ['2023-01-01 00:00:00'],
        'patientId': [3],
        'temperatureAirSensor': [22.0]
    })

    monkeypatch.setattr('back_inference.feature_engineering', lambda **kwargs: pd.DataFrame({'patientId':[3], 'sex':['H'], 'feature1':[1]}))
    df_out = preparer_df(df)
    assert df_out['sex'].iloc[0] == 'H'
