import pandas as pd
from joblib import load 
from flask import request, jsonify, render_template
from app import app
import lightgbm as lgb

# Charger le DataFrame
test_df = pd.read_csv("./Assets/test_w2_df.csv")
feats = [f for f in test_df.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index',"IF_0_CREDIT_IS_OKAY","PAYBACK_PROBA",'CODE_GENDER']]
modele = load("./Assets/Lgb_w2.joblib")
df=test_df[feats]
df=df.iloc[:1000]
@app.route('/')
def home():
    ids = df["SK_ID_CURR"].unique().tolist() 
    return render_template("select_id.html", ids=ids)

@app.route('/predict', methods=['GET'])
def predict():
    id_client = request.args.get('id', default=None, type=int)

    if id_client not in df['SK_ID_CURR'].values:
        return jsonify({"erreur": "ID non trouvé"}), 404

    # Sélectionner la ligne correspondant à l'ID
    ligne_client = df[df['SK_ID_CURR'] == id_client].drop(columns=['SK_ID_CURR'])

    # Prédiction
    proba = modele.predict_proba(ligne_client)[0]
    proba_dict = {f"Classe {i}": prob for i, prob in enumerate(proba, start=1)}

    # Pour simplifier, retourner le résultat comme texte brut ou JSON
    return jsonify(proba_dict)