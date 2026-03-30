import csv
import json
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
AUDIT_DIR = Path("audit_trail")
COLONNES = ["timestamp", "utilisateur", "module", "action", "fichiers", "scores", "statut", "message"]

# --- MODULES ---
MODULE_SYSTEME     = "systeme"
MODULE_DONNEES     = "donnees"
MODULE_INFEREUR    = "infereur"
MODULE_COMPARATEUR = "comparateur"
MODULE_RAPPORT     = "rapport"

# --- ACTIONS ---
ACTION_OUVERTURE_SESSION           = "OUVERTURE_SESSION"
ACTION_APPLICATION_FILTRES         = "APPLICATION_FILTRES"
ACTION_LANCEMENT_DOUBLE_INFERENCE  = "LANCEMENT_DOUBLE_INFERENCE"
ACTION_COMPARAISON_GOLD            = "COMPARAISON_GOLD"
ACTION_GENERATION_RAPPORT_PDF      = "GENERATION_RAPPORT_PDF"
ACTION_ERREUR_INFERENCE            = "ERREUR_INFERENCE"


# --- STATUTS ---
STATUT_SUCCESS       = "SUCCESS"
STATUT_ERREUR        = "ERREUR"

def nettoyer_username(username: str) -> str:
    import unicodedata
    norme = unicodedata.normalize("NFD", username)
    ascii_ = norme.encode("ascii", "ignore").decode("ascii")
    propre = "".join(c if c.isalnum() else "_" for c in ascii_)
    propre = propre.lower().strip("_")
    return propre or "anonyme"

def initialiser_session(username: str) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    nom_propre = nettoyer_username(username)
    chemin = AUDIT_DIR / f"{horodatage}_{nom_propre}.csv"

    with open(chemin, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES)
        writer.writeheader()

    log_action(chemin, MODULE_SYSTEME, ACTION_OUVERTURE_SESSION, username, message="Connexion reussie")
    return chemin

def log_action(chemin_fichier: Path, module: str, action: str, utilisateur: str, fichiers: list = None, scores: dict = None, statut: str = STATUT_SUCCESS, message: str = "", exc: Exception = None):
    if exc:
        message = f"{message} | {type(exc).__name__}: {exc}"
        statut = STATUT_ERREUR

    utilisateur_propre = nettoyer_username(utilisateur)

    ligne = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "utilisateur": utilisateur_propre,
        "module": module,
        "action": action,
        "fichiers": json.dumps(fichiers or [], ensure_ascii=False),
        "scores": json.dumps(scores or {}, ensure_ascii=False),
        "statut": statut,
        "message": message,
    }
    try:
        with open(chemin_fichier, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLONNES)
            writer.writerow(ligne)
    except OSError:
        pass 