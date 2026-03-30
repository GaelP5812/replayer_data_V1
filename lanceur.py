import streamlit.web.cli as stcli
import sys
import os
import pandas
import xgboost
import comparateur_back 
import back_inference   
import back_filtre
import audit_trail_back

if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        # 1. Le dossier temporaire secret où ton code (front.py) est caché
        dossier_app = sys._MEIPASS
        
        dossier_physique = os.path.dirname(os.path.abspath(sys.argv[0]))
        
        os.chdir(dossier_physique)
    else:
        dossier_app = os.path.dirname(os.path.abspath(__file__))
        os.chdir(dossier_app)
    
    chemin_front = os.path.join(dossier_app, "front.py")
    
    # Lancement de Streamlit
    sys.argv = ["streamlit", "run", chemin_front, "--global.developmentMode=false"]
    sys.exit(stcli.main())