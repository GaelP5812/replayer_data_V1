import pytest
import pandas as pd
import os

from back_filtre import charger_brassards 

def test_charger_brassards_dossier_vide(tmp_path):
    """Test le cas où aucun fichier n'existe dans le dossier."""
    # tmp_path est un objet Path, on le convertit en string pour ta fonction
    df = charger_brassards(dossier=str(tmp_path))
    
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_charger_brassards_un_fichier(tmp_path):
    """Test avec un seul fichier valide (ex: data1.csv)."""
    fichier = tmp_path / "data1.csv"
    fichier.write_text("id,valeur\n1,A\n2,B") # Création d'un faux CSV

    df = charger_brassards(dossier=str(tmp_path))
    
    assert not df.empty
    assert len(df) == 2
    assert list(df["valeur"]) == ["A", "B"]

def test_charger_brassards_plusieurs_fichiers_concat(tmp_path):
    """Test la concaténation de plusieurs fichiers valides."""
    (tmp_path / "data1.csv").write_text("id,valeur\n1,A")
    (tmp_path / "data5.csv").write_text("id,valeur\n2,B")
    (tmp_path / "data29.csv").write_text("id,valeur\n3,C")

    df = charger_brassards(dossier=str(tmp_path))
    
    assert len(df) == 3
    # Vérifie que ignore_index=True a bien fonctionné
    assert list(df.index) == [0, 1, 2] 

def test_charger_brassards_ignore_mauvais_noms(tmp_path):
    """Test que la fonction ignore les fichiers hors de la plage 1-29 ou mal nommés."""
    (tmp_path / "data1.csv").write_text("id,valeur\n1,A")
    # Ces fichiers ne doivent PAS être lus par la fonction
    (tmp_path / "data30.csv").write_text("id,valeur\n99,Z")
    (tmp_path / "autre_fichier.csv").write_text("id,valeur\n99,Z")

    df = charger_brassards(dossier=str(tmp_path))
    
    # On ne doit retrouver que la ligne du fichier data1.csv
    assert len(df) == 1
    assert df.iloc[0]["valeur"] == "A"