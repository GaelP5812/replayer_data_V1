import streamlit as st
import os
import pandas as pd
from back_inference import executer_double_inference
from comparateur_back import evaluer_modeles_globaux, generer_conclusion_texte, generer_rapport_pdf
from audit_trail_back import (initialiser_session, log_action, MODULE_DONNEES, MODULE_INFEREUR, MODULE_COMPARATEUR, MODULE_RAPPORT,
    ACTION_APPLICATION_FILTRES, ACTION_LANCEMENT_DOUBLE_INFERENCE, 
    ACTION_COMPARAISON_GOLD, ACTION_GENERATION_RAPPORT_PDF, ACTION_ERREUR_INFERENCE, STATUT_ERREUR
)
from datetime import datetime

# =============================================================


#                        PARTIE FILTRE


# =============================================================

if "username" not in st.session_state:
    st.set_page_config(page_title="Replayer", layout="centered")
    st.title("Replayer")
    st.info("Cette application requiert une identification pour la traçabilité des données.")

    with st.form("login_form"):
        nom_saisi = st.text_input("Veuillez saisir votre Prénom :")
        submitted = st.form_submit_button("Se connecter", type="primary")

    if submitted:
        if nom_saisi.strip():
            st.session_state["username"] = nom_saisi.strip()
            chemin_audit = initialiser_session(nom_saisi.strip())
            st.session_state["audit_file"] = chemin_audit
            st.rerun()
        else:
            st.error("Le nom est obligatoire pour accéder à l'application.")

    st.stop()

if st.session_state.get("reinitialiser_filtres"):
    st.session_state["filtre_sexe_H"] = True
    st.session_state["filtre_sexe_F"] = True
    st.session_state["filtre_poids_min"] = 30
    st.session_state["filtre_poids_max"] = 120
    st.session_state["filtre_taille_min"] = 0
    st.session_state["filtre_taille_max"] = 200
    st.session_state["filtre_age_min"] = 18 
    st.session_state["filtre_age_max"] = 100
    st.session_state["reinitialiser_filtres"] = False

elif "filtre_sexe_H" not in st.session_state:
    st.session_state["filtre_sexe_H"] = True
    st.session_state["filtre_sexe_F"] = True
    st.session_state["filtre_poids_min"] = 30
    st.session_state["filtre_poids_max"] = 120
    st.session_state["filtre_taille_min"] = 0
    st.session_state["filtre_taille_max"] = 200
    st.session_state["filtre_age_min"] = 18
    st.session_state["filtre_age_max"] = 100

def afficher_filtre(df):
    """
    Affiche le panneau de filtres démographiques.
    Retourne le df filtré et un booléen indiquant si l'utilisateur a confirmé.
    """

    st.markdown("---")

    # Bouton sélectionner toutes les données
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        if st.button("Sélectionner toutes les données", use_container_width=True):
            st.session_state["filtres_confirmes"] = True
            st.session_state["df_filtre"] = df
            st.session_state["texte_filtres_sauves"] = "Aucun (Toutes les donnees)"
            st.session_state["filtre_sexe_H"] = True
            st.session_state["filtre_sexe_F"] = True
            st.session_state["filtre_poids_min"] = None
            st.session_state["filtre_poids_max"] = None
            st.session_state["filtre_taille_min"] = None
            st.session_state["filtre_taille_max"] = None
            st.session_state["filtre_age_min"] = None
            st.session_state["filtre_age_max"] = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Ligne 1 : Sexe et Poids
    col_sexe, col_poids = st.columns(2)

    with col_sexe:
        st.markdown("**Sexe**")
        sexe_H = st.checkbox("Homme", key="filtre_sexe_H")
        sexe_F = st.checkbox("Femme", key="filtre_sexe_F")

    with col_poids:
        st.markdown("**Poids (kg)**")
        col_pmin, col_pmax = st.columns(2)
        with col_pmin:
            poids_min = st.number_input("Min", min_value=30, max_value=130, step=1, key="filtre_poids_min")
        with col_pmax:
            poids_max = st.number_input("Max", min_value=30, max_value=130, step=1, key="filtre_poids_max")

    st.markdown("<br>", unsafe_allow_html=True)

        # Ligne 2 : Taille et Âge
    col_taille, col_age = st.columns(2)

    with col_taille:
        st.markdown("**Taille (cm)**")
        col_tmin, col_tmax = st.columns(2)
        with col_tmin:
            taille_min = st.number_input("Min", min_value=0, max_value=200, step=1, key="filtre_taille_min")
        with col_tmax:
            taille_max = st.number_input("Max", min_value=0, max_value=200, step=1, key="filtre_taille_max")

    with col_age:
        st.markdown("**Âge (ans)**")
        col_amin, col_amax = st.columns(2)
        with col_amin:
            age_min = st.number_input("Min", min_value=18, max_value=100, step=1, key="filtre_age_min")
        with col_amax:
            age_max = st.number_input("Max", min_value=18, max_value=100, step=1, key="filtre_age_max")

    st.markdown("<br>", unsafe_allow_html=True)

    # Compteur données actuelles — grand et visible
    df_filtre = _appliquer_filtres_live(df, sexe_H, sexe_F, poids_min, poids_max, taille_min, taille_max, age_min, age_max)

    st.markdown(
        f"""
        <div style="
            background-color: #1a1a2e;
            border: 2px solid #4a90d9;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <span style="font-size: 1rem; color: #a0b4c8;">Données correspondantes</span><br>
            <span style="font-size: 2.8rem; font-weight: bold; color: #4a90d9;">{len(df_filtre):,}</span>
            <span style="font-size: 1.2rem; color: #a0b4c8;"> lignes</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)




    # Boutons Réinitialiser | Confirmer
    col_reinit, col_confirmer = st.columns(2)

    confirme = False

    with col_reinit:
        if st.button("🔄 Réinitialiser les filtres", use_container_width=True):
            st.session_state["reinitialiser_filtres"] = True
            st.rerun()

    with col_confirmer:
        if st.button("✅ Confirmer la sélection", use_container_width=True, type="primary"):
            if len(df_filtre) == 0:
                st.error("Aucune donnée ne correspond aux critères sélectionnés.")
            else:
                st.session_state["filtres_confirmes"] = True
                st.session_state["df_filtre"] = df_filtre

                filtres_actifs = []
                if not (sexe_H and sexe_F):
                    if sexe_H: filtres_actifs.append("Sexe: M")
                    elif sexe_F: filtres_actifs.append("Sexe: F")
                
                if poids_min > 30 or poids_max < 120:
                    filtres_actifs.append(f"Poids: {poids_min}-{poids_max}kg")
                    
                if taille_min > 0 or taille_max < 200:
                    filtres_actifs.append(f"Taille: {taille_min}-{taille_max}cm")
                    
                if age_min > 18 or age_max < 100:
                    filtres_actifs.append(f"Age: {age_min}-{age_max}ans")

                if not filtres_actifs:
                    st.session_state["texte_filtres_sauves"] = "Aucun (Toutes les donnees)"
                else:
                    st.session_state["texte_filtres_sauves"] = " | ".join(filtres_actifs)

                log_action(
                    chemin_fichier=st.session_state["audit_file"],
                    module=MODULE_DONNEES,
                    action=ACTION_APPLICATION_FILTRES,
                    utilisateur=st.session_state["username"],
                    message=f"Filtres appliques : {st.session_state['texte_filtres_sauves']} (Volume restant : {len(df_filtre)})"
                )
                
                st.rerun()

    st.markdown("---")

    return df_filtre, confirme


def _appliquer_filtres_live(df, sexe_H, sexe_F, poids_min, poids_max, taille_min, taille_max, age_min, age_max):
    """Applique les filtres en temps réel pour le compteur."""
    sexes = []
    if sexe_H:
        sexes.append("M")
    if sexe_F:
        sexes.append("F")

    df_f = df.copy()

    if sexes:
        df_f = df_f[df_f["sex"].isin(sexes)]
    else:
        df_f = df_f.iloc[0:0]  

    df_f = df_f[(df_f["weight"] >= poids_min) & (df_f["weight"] <= poids_max)]
    df_f = df_f[(df_f["height"] >= taille_min) & (df_f["height"] <= taille_max)]
    df_f = df_f[(df_f["age"] >= age_min) & (df_f["age"] <= age_max)]

    return df_f


# =============================================================


#                        PARTIE ALGO


# =============================================================

def afficher_page_modeles(df_filtre):
    st.title("Replayer")
    st.metric(label="Données prêtes pour l'inférence", value=f"{len(df_filtre):,} lignes")
    st.markdown("---")

    nom_comparaison = st.text_input("Nom de la comparaison :", value="Comparaison_Standard")
    
    upload_key = st.session_state.get("upload_key", 0)

    col_A, col_B = st.columns(2)
    with col_A:
        fichier_A = st.file_uploader("Glissez l'ancien algorithme (.sav)", type=['sav'], key=f"algo_A_{upload_key}")
    with col_B:
        fichier_B = st.file_uploader("Glissez le nouvel algorithme (.sav)", type=['sav'], key=f"algo_B_{upload_key}")

    if fichier_A is not None and fichier_B is not None:
        st.success("✅ Algorithmes chargés !")
        st.markdown("---")

        if (st.session_state.get("nom_fichier_A") != fichier_A.name or
            st.session_state.get("nom_fichier_B") != fichier_B.name):
            for key in ["df_final", "pdf_data", "scores_A", "scores_B", "conclusion"]:
                st.session_state.pop(key, None)
            st.session_state["nom_fichier_A"] = fichier_A.name
            st.session_state["nom_fichier_B"] = fichier_B.name
        
        if "df_final" in st.session_state:

            if "pdf_data" not in st.session_state:
                with st.spinner("Analyse et génération du rapport PDF..."):
                    try:
                        df_merge, scores_A, scores_B = evaluer_modeles_globaux(
                            st.session_state["df_final"]
                        )

                        log_action(
                            chemin_fichier=st.session_state["audit_file"],
                            module=MODULE_COMPARATEUR,
                            action=ACTION_COMPARAISON_GOLD,
                            utilisateur=st.session_state["username"],
                            scores={"Nouvel Algo": scores_B['score_agg'], "Ancien Algo": scores_A['score_agg']},
                            message=f"Comparaison reussie sur {len(df_merge)} minutes communes."
                        )

                        conclusion = generer_conclusion_texte(scores_A, scores_B)
                        texte_filtres = st.session_state.get("texte_filtres_sauves", "Aucun (Toutes les donnees)")
                        volume_lignes = f"{len(df_merge):,}".replace(',', ' ')

                        pdf_data = generer_rapport_pdf(
                            nom_comparaison=nom_comparaison,
                            nom_utilisateur=st.session_state["username"],
                            texte_filtres=texte_filtres,
                            volume_lignes=volume_lignes,
                            nom_fichier_A=fichier_A.name,
                            nom_fichier_B=fichier_B.name,
                            scores_A=scores_A,
                            scores_B=scores_B,
                            conclusion=conclusion
                        )

                        log_action(
                            chemin_fichier=st.session_state["audit_file"],
                            module=MODULE_RAPPORT,
                            action=ACTION_GENERATION_RAPPORT_PDF,
                            utilisateur=st.session_state["username"],
                            fichiers=[fichier_A.name, fichier_B.name],
                            message=f"PDF genere pour le projet : {nom_comparaison}"
                        )

                        st.session_state["pdf_data"] = pdf_data
                        st.session_state["scores_A"] = scores_A
                        st.session_state["scores_B"] = scores_B
                        st.session_state["conclusion"] = conclusion

                    except Exception as e:
                        st.error(f"Erreur : {e}")

            # Affichage écran
            if "scores_A" in st.session_state and "scores_B" in st.session_state:
                scores_A = st.session_state["scores_A"]
                scores_B = st.session_state["scores_B"]
                conclusion = st.session_state["conclusion"]

                col_met_A, col_vs, col_met_B = st.columns([2, 1, 2])
                with col_met_A:
                    st.info("🔵 **Ancien Algo**")
                    st.metric("Score", f"{scores_A['score_agg']:.2f}")
                with col_vs:
                    st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
                with col_met_B:
                    st.success("🟢 **Nouvel Algo**")
                    st.metric("Score", f"{scores_B['score_agg']:.2f}", delta=f"{scores_B['score_agg'] - scores_A['score_agg']:.2f}")

                st.write(f"**Conclusion :** {conclusion}")
                st.divider()

                date_str_file = datetime.now().strftime("%Y%m%d_%H%M")
                nom_fichier_propre = nom_comparaison.replace(" ", "_")
                nom_final_pdf = f"compte_rendu_{nom_fichier_propre}_{date_str_file}.pdf"

                # Bouton de téléchargement 
                st.download_button(
                    label="📄 Télécharger le compte rendu (PDF)",
                    data=st.session_state["pdf_data"],
                    file_name=nom_final_pdf,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )

                st.write("")
                
                col_filtres, col_nouveau = st.columns(2)

                with col_filtres:
                    if st.button("🔧 Changer les filtres", use_container_width=True):
                        for key in ["filtres_confirmes", "df_filtre", "df_final", "pdf_data", "scores_A", "scores_B", "conclusion"]:
                            st.session_state.pop(key, None)
                        st.rerun()

                with col_nouveau:
                    if st.button("🔄 Refaire un nouveau test", use_container_width=True):
                        for key in ["df_final", "pdf_data", "scores_A", "scores_B", "conclusion"]:
                            st.session_state.pop(key, None)
                        st.session_state["upload_key"] = st.session_state.get("upload_key", 0) + 1
                        st.rerun()

        else:
            _, col_btn, _ = st.columns([1, 2, 1])
            with col_btn:
                if not nom_comparaison.strip():
                    st.warning("Veuillez saisir un nom pour la comparaison.")
                elif st.button("Lancer la comparaison", use_container_width=True):
                    with st.spinner("⏳ Inférence en cours..."):

                        log_action(
                            chemin_fichier=st.session_state["audit_file"],
                            module=MODULE_INFEREUR,
                            action=ACTION_LANCEMENT_DOUBLE_INFERENCE,
                            utilisateur=st.session_state["username"],
                            fichiers=[fichier_A.name, fichier_B.name],
                            message="Lancement de la double inference sur les modeles .sav"
                        )
                        try:
                            df_resultat = executer_double_inference(df_filtre, fichier_A, fichier_B)
                            st.session_state["df_final"] = df_resultat
                            st.rerun()
                        except ValueError as e:
                            st.error(f"❌ {e}")
                            log_action(
                                chemin_fichier=st.session_state["audit_file"],
                                module=MODULE_INFEREUR,
                                action=ACTION_ERREUR_INFERENCE,
                                utilisateur=st.session_state["username"],
                                fichiers=[fichier_A.name, fichier_B.name],
                                statut=STATUT_ERREUR,
                                message=str(e)
                            )
                        except Exception as e:
                            st.error(f"❌ Erreur inattendue : {e}")
                            log_action(
                                chemin_fichier=st.session_state["audit_file"],
                                module=MODULE_INFEREUR,
                                action=ACTION_ERREUR_INFERENCE,
                                utilisateur=st.session_state["username"],
                                fichiers=[fichier_A.name, fichier_B.name],
                                statut=STATUT_ERREUR,

                                message=f"Erreur inattendue : {e}"
                            )
    else:
        st.info("💡 Veuillez glisser un fichier .sav dans chaque zone.")

# =============================================================


#                        MAIN


# =============================================================


if __name__ == "__main__":
    dfs = []
    for i in range(1, 30):
        chemin = os.path.join("data", f"data{i}.csv")
        if os.path.exists(chemin):
            dfs.append(pd.read_csv(chemin, sep=None, engine='python'))
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Filtre
    if not st.session_state.get("filtres_confirmes"):
        df_filtre, confirme = afficher_filtre(df)
        if confirme:
            st.session_state["filtres_confirmes"] = True
            st.session_state["df_filtre"] = df_filtre
            st.rerun()

    # Comparaison
    else:
        if st.button("← Retour aux filtres"):
            for key in ["filtres_confirmes", "df_filtre", "df_final", "pdf_data", "scores_A", "scores_B", "conclusion"]:
                st.session_state.pop(key, None)
            st.rerun()
        afficher_page_modeles(st.session_state["df_filtre"])