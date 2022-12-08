# source :
# https://medium.com/@kaustuv.kunal/how-to-deploy-and-host-machine-learning-model-de8cfe4de9c5
# https://github.com/kaustuvkunal/BasicMLModelFlaskDeploy
# customizé pour fusion avec webinaire
# 
# environnement : env_app
# flask  2.2.2
# joblib  1.2.0
# numpy  1.21.6
# requests 2.28.1
# scikit-learn==1.0.2 => sklearn est utilisé à travers la fonction joblib
# shap ...

# python app.py

from flask import Flask, render_template, request, redirect, url_for, json
import joblib
import pickle
import numpy as np
from requests.models import Response
import shap
import time

app = Flask(__name__)

@app.route("/")
def root():
    return render_template("index.html")

# @app.route("/loading_model", methods=['GET'])
# def load_model():
#     print("========== loading model ==============")
#     loaded_model = joblib.load('model_lgbm.joblib')
#     print("========== model loaded ==============")
    
#     # on affiche le résultat sur la page index.html avec la méthode GET
#     msg = "========== model loaded =============="
#     return render_template("index.html", text_from_app=msg)


# @app.route("/init", methods=['GET'])
# def init():  
#     df = pd.read_csv('dataset/data_for_mini_model.csv', nrows=6)
#     msg = "========== init data =============="
#     return render_template("index.html", text= msg)
#     print("========== init data ==============")
#     response = app.response_class(
#         response=json.dumps([df]),
#         status=200,
#         mimetype='application/json'
#     )
#     return response
    # return make_response(jsonify(df), 200, mimetype='application/json')

@app.route("/predict", methods=['GET', 'POST'])
def make_prediction():

    print("========== loading model ==============")
    loaded_model = joblib.load('model_lgbm.joblib')
    print("========== model loaded ==============")
    
    # # on affiche le résultat sur la page index.html avec la méthode GET
    # msg = "========== model loaded =============="
    # return render_template("index.html", text_from_app=msg)
    
    
    proba = np.nan
    print("============= request method : {}=============".format(request.method))

    if request.method=='POST':
        # Pour lancer une requête POST depuis l'invite de commande, taper dans l'invite de commande :
        # curl http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"data\": [[1, 2, 3, 4, 5, 6, 7, 8]]}"
        data = request.get_json()
        # print("============= data: {}=============".format(data['data']))
        print("============= data shape: {}=============".format(np.shape(data['data'])))
        # print("============= data : {}=============".format(request.form['data']))
        X = np.array([float(var) for var in data['data'][0]])
    elif request.method=='GET':
        # Pour transmettre les arguments avec la méthode GET on passe par la barre d'adresse :
        # http://localhost:5000/predict?ARGUMENTS
        # Les ARGUMENTS sont séparés par des & tel que ci-dessous:
        # revenu_med=3.87&age_med=28.0&nb_piece_med=5.0&nb_chambre_moy=1.0&taille_pop=1425&occupation_moy=3.0&latitude=35.0&longitude=-119.0
        data = request.args
        X = np.array([float(var) for var in data.values()])
    
    print("X shape = {}".format(X.shape))
    if X.reshape(1, -1).shape != (1,732):
        raise ValueError("X has not (1,732) size")
    
    # print("============= X : {}=============".format(X))
    proba = loaded_model.predict_proba(X.reshape(1, -1))[0][1]
    # print("============= prediction : {}=============".format(proba))
    
    
    proba = proba.round(2)
    print("============= prediction : {}=============".format([proba]))  

    shap_wanted=False
    if shap_wanted:
        explainer = shap.TreeExplainer(loaded_model)
        start = time.time()
        shap_values = explainer.shap_values(df0[features_names].values)
        time_shapvalues = time.time()-start
        print("shapvalues computing time : {}" .format(time_shapvalues))
      
    # On renvoie une réponse à l'aide de la classe response_class de flask
    response = app.response_class(
        response=json.dumps([proba]),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    app.run(debug=True)
