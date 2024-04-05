import pandas as pd
from joblib import load 
from flask import request, jsonify, render_template
from app import app
import lightgbm as lgb

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

except Exception as e:
    print(f"Une erreur s'est produite: {e}")
    traceback.print_exc()  # Imprime la pile d'appels pour aider au diagnostic





# Charger le DataFrame
#test_df = pd.read_csv("./Assets/test_w2_df.csv")
feats = [f for f in test_df.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index',"IF_0_CREDIT_IS_OKAY","PAYBACK_PROBA",'CODE_GENDER']]
#modele = load("./Assets/Lgb_w2.joblib")
df=test_df[feats]
df=df.iloc[:1000]


ligne_client = df[df['SK_ID_CURR'] == 100001].drop(columns=['SK_ID_CURR'])

    # Prédiction
print(modele.predict_proba(ligne_client)[0])


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