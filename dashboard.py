import pandas as pd
import streamlit as st
import requests

# requests 2.28.1
# streamlit 1.14.0
# pandas 1.3.5

# streamlit run dashboard.py

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    print("===================data_json = {} ".format(data_json))
    response = requests.request(method='POST', 
                                headers=headers, url=model_uri, json=data_json)

    # response = requests.request(method='GET', 
    #                             headers=headers, url=model_uri, json=data_json)
    
    print("================================== response: {}, {}===================".format(response.status_code, response.text))
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    # FLASK_URI = 'http://projet7a-py0153.pythonanywhere.com/predict'
    FLASK_URI = 'http://127.0.0.1:5000/predict'

    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['Flask', 'Cortex', 'Ray Serve'])

    st.title('Failure risk estimation for loan delivery')

    print("Lecture de la base de données")
    df = pd.read_csv('dataset/featureengineeringdata.csv', nrow=5)
    
    revenu_med = st.number_input('Revenu médian dans le secteur (en 10K de dollars)',
                                 min_value=0., value=3.87, step=1.)

    age_med = st.number_input('Âge médian des maisons dans le secteur',
                              min_value=0., value=28., step=1.)

    nb_piece_med = st.number_input('Nombre moyen de pièces',
                                   min_value=0., value=5., step=1.)

    nb_chambre_moy = st.number_input('Nombre moyen de chambres',
                                     min_value=0., value=1., step=1.)

    taille_pop = st.number_input('Taille de la population dans le secteur',
                                 min_value=0, value=1425, step=100)

    occupation_moy = st.number_input('Occupation moyenne de la maison (en nombre d\'habitants)',
                                     min_value=0., value=3., step=1.)

    latitude = st.number_input('Latitude du secteur',
                               value=35., step=1.)

    longitude = st.number_input('Longitude du secteur',
                                value=-119., step=1.)

    sample_index = st.number_input("Index du client",
                                min_value=0, value=5, step=1)
    
    predict_btn = st.button('Prédire')
    if predict_btn:
        data = [df.iloc[sample_index].values]
        # data = [[revenu_med, age_med, nb_piece_med, nb_chambre_moy,
        #          taille_pop, occupation_moy, latitude, longitude]]
        print("===================data = {} ".format(data))
        print("===================data = {} ".format(np.shape(data)))
        # [[3.87, 28.0, 5.0, 1.0, 1425, 3.0, 35.0, -119.0]]
        
        pred = None

        if api_choice == 'Flask':
            pred = request_prediction(FLASK_URI, data)
        elif api_choice == 'Cortex':
            pred = request_prediction(CORTEX_URI, data)[0] * 100000
        elif api_choice == 'Ray Serve':
            pred = request_prediction(RAY_SERVE_URI, data)[0] * 100000
        st.write(
            "La proba d'une défaillance du client est de {:.2f}".format(pred))

if __name__ == '__main__':
    main()
