import pandas as pd
from joblib import load 
from flask import request, jsonify, render_template
from app import app
import lightgbm as lgb
import dill 
import lime 
import lime.lime_tabular
import numpy as np
################
import io
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import traceback

################################
account_name = "fruitsamurai97depot"
account_key=''
with open("azure_container_key.txt", "r") as my_key:
    account_key= my_key.read()
container_name= "assets"
################################
try:
    with open("azure_container_key.txt", "r") as my_key:
        account_key= my_key.read().strip()

    connect_str = f'DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net'
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    # Chargement du DataFrame depuis le blob
    test_df_name = "test_w2_df.csv"
    sas_test = generate_blob_sas(account_name=account_name,
                                container_name=container_name,
                                blob_name=test_df_name,
                                account_key=account_key,
                                permission=BlobSasPermissions(read=True),
                                expiry=datetime.utcnow() + timedelta(hours=1))
    sas_test_url = f'https://{account_name}.blob.core.windows.net/{container_name}/{test_df_name}?{sas_test}'
    test_df = pd.read_csv(sas_test_url)
    # Chargement du modèle depuis le blob
    model_blob_name = "Lgb_w2.joblib"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_blob_name)
    stream = io.BytesIO()
    blob_client.download_blob().download_to_stream(stream)
    stream.seek(0)  # Retour au début du stream
    modele = load(stream)
    

    explainer_blob_name = 'lime_explainer_w2.pkl'
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=explainer_blob_name)
    stream = io.BytesIO()
    blob_client.download_blob().download_to_stream(stream)
    stream.seek(0)  # Réinitialise le pointeur au début du stream pour la lecture
    explainer = dill.load(stream) 


except Exception as e:
    print(f"Une erreur s'est produite: {e}")
    traceback.print_exc()  # Imprime la pile d'appels pour aider au diagnostic


feats = [f for f in test_df.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index',"IF_0_CREDIT_IS_OKAY","PAYBACK_PROBA",'CODE_GENDER']]
df=test_df[feats]
df=df.iloc[:1000]


@app.route('/client',methods=["GET"])
def get_client():


    ids = df["SK_ID_CURR"].unique().tolist() 
    return jsonify(ids)#render_template("select_id.html", ids=ids)

@app.route('/predict', methods=['GET'])
def predict():

    #global modele  # Ajoutez cette ligne pour indiquer que modele est une variable globale

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

@app.route("/explain",methods=["GET"])
def explain():
    id_client = request.args.get('id', default=None, type=int)

    if id_client not in df['SK_ID_CURR'].values:
        return jsonify({"erreur": "ID non trouvé"}), 404

    # Sélectionner la ligne correspondant à l'ID
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index',"IF_0_CREDIT_IS_OKAY","PAYBACK_PROBA",'CODE_GENDER']]
    ligne_client = df[df['SK_ID_CURR'] == id_client]
    ligne_client = ligne_client[feats].iloc[0]
    client_instance = np.array(ligne_client)
    exp= explainer.explain_instance(
        data_row=client_instance, 
        predict_fn=modele.predict_proba, 
        num_features=5
    )
    return jsonify(exp.as_list())


@app.route("/info",methods=["GET"])
def get_info():
    id_client = request.args.get('id', default=None, type=int)
    if id_client not in df['SK_ID_CURR'].values:
        return jsonify({"erreur": "ID non trouvé"}), 404
    col_sel=["CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH"]
    ligne_client = df[df['SK_ID_CURR'] == id_client]
    info_client = ligne_client[col_sel].iloc[0]

    return jsonify(info_client)

