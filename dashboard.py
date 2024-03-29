import pandas as pd
import streamlit as st
import requests
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import plotly.figure_factory as ff
import plotly.express as px
import sys

# requests 2.28.1
# streamlit 1.14.0
# streamlit-aggrid 0.3.2 (installer avec Pip)
# plotly 5.11.0 (conda install -c plotly plotly=5.11.0)
# pandas 1.3.5
sys.stdout.flush()

# streamlit run dashboard.py

# create an interactive table
def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.SELECTION_CHANGED, # MODEL_CHANGED,
        allow_unsafe_jscode=False, # True
    )

    return selection


def request_prediction(model_uri, data, features_names):
    headers = {"Content-Type": "application/json"}
  
    print("============== data.shape :", np.shape(data))
    print("===================data type = ", type(data))
    data_json = {'data': data, 'features_names': features_names}
    print("===================data_json type = ", type(data_json))
    print("predict url : ", model_uri)
    response = requests.request(method='POST', 
                                headers=headers, url=model_uri, json=data_json)

    # response = requests.request(method='GET', 
    #                             headers=headers, url=model_uri, json=data_json)
    
 
    print("=============== response type: {}==================="
          .format(type(response.json())))
    if response.status_code != 200:
        raise Exception("Request failed with status {}, {}"
                        .format(response.status_code, response.text))
    # print("response.text : ", response.text) # contenu de la réponse en format text
    print("response.json : ", response.json()) # contenu de la réponse en format json
    return response.json()

def display_result(pred):
        # Transforme la liste de pred["shap_values"] en valeur numérique
        pred["shap_values"] = [round(float(value),3) for value in pred["shap_values"]]
        
        st.write("La proba d'une défaillance du client est de {:.2f}"
                 .format(pred["proba"]))
        print("=== Les features plus importantes lors de cette prédiction sont : === {}"
                .format(pred["features"]))
        print("=== Les shapley_values correspondantes sont === {} "
                .format(pred["shap_values"]))

        features_importances = pd.DataFrame(pred["shap_values"], index=pred["features"])
        features_importances["influence"]=["positive" if value>0 
                                        else "negative" 
                                        for value in pred["shap_values"]]
        # define color for negative and positive value in the graph
        red = px.colors.qualitative.Set1[0]
        blue = px.colors.qualitative.Set1[1]
        color = [ [blue, red] if pred["shap_values"][0]>0 else [red, blue] ]
        
        # Comportement étrange: color_discrete_sequence doit avoir la dimension de 
        #                       features_importances["influence"] mais plotly ne prend
        #                       que les 2 premières comme si il affectait ces valeurs
        #                       à "negative" et "positive".
        color_discrete_sequence = [None]*10
        color_discrete_sequence[:2] = color[0]
        
        fig = px.bar(features_importances,
                     title="Importance des features dans la prédiction de la probabilité de défaillance",
                     labels={"index":"Features Names", "value": "Features Importances"},
                            #"variable": "SHAPley values"},
                     color ="influence",
                     color_discrete_sequence=color_discrete_sequence)
        # pour afficher les valeurs tels qu'elles sont dans le df (sinon px s'amuse à les réordonnancer)
        fig.update_layout(xaxis={'categoryorder': "array",
                                 'categoryarray': 
                                         [str(i) for i in features_importances.index]})
        st.plotly_chart(fig)
        path="dataset/"
        df_tmp = pd.read_csv(path+'HomeCredit_columns_description.csv', encoding_errors='ignore')
        features_description=dict()
        for feature_name in pred["features"]:
            if feature_name in df_tmp.Row.values:
                features_description[feature_name] = df_tmp[df_tmp.Row==feature_name].Description.values[0]
                
        st.write(features_description)
                
#############
# @st.cache
# def load_model():
# 	  return torch.load("path/to/model.pt")

def main():
    # FLASK_URI = 'http://projet7-py0153.pythonanywhere.com/predict'
    # FLASK_URI = 'http://127.0.0.1:5000/predict'
    
    # response = requests.request(method='GET',  url=LOAD_MODEL_URI)
    # print("===================model loaded =====")
    
    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ('pythonanywhere', 'local'))
    if api_choice=="pythonanywhere": 
        FLASK_URI='http://projet7-py0153.pythonanywhere.com/predict'
    elif api_choice=="local": 
        FLASK_URI='http://127.0.0.1:5000/predict'
    else :
        raise(error)
    
    st.title('Failure risk estimation for loan delivery')

    print("Lecture de la base de données")
    df = pd.read_csv('dataset/df_sub_01.csv') # nrows=6)
    features_names = df.columns.tolist()
    
    # options = st.sidebar.multiselect(
    # 'Par quelle feature souhaitez vous filtrer ?',
    # df.columns, 'CNT_CHILDREN')

    options = st.sidebar.selectbox(
    'Par quelle feature souhaitez vous filtrer ?',
    df.columns, 0)
    
    st.write('You selected:', options)
    # filter_btn = st.button('Filtrer')
    
    if options:

        x = df[options].values
        x_min = float(x.min())
        x_max = float(x.max())

        plot_spot = st.empty()  # une autre alternative serait d'utiliser les containers

        values = st.slider('Select a range of values',
                                x_min, x_max, (x_min, x_max)) 
        # st.write('values:', values)
        
        fig = px.histogram(df[options][(df[options]>=values[0]) & (df[options]<=values[1])])
        
        with plot_spot:
            st.plotly_chart(fig) #
        
        # réduction du nombre de lignes, au-dessus de 200 MB 
        # (streamlit+pandas) utilise environ 4 fois la taille du tableau en mémoire 
        number_of_rows = ((df[options]>=values[0]) & (df[options]<=values[1])).sum()
        # number_of_rows = ((df[options[0]]>=values[0]) & (df[options[0]]<=values[1])).sum()
        st.write('Number of rows:', number_of_rows )  
        
        # proceed_btn = st.button('Proceed')
        
        selection=[]
        if number_of_rows<1000 :  # En théorie 5000 ça passe mais ça ramme sur le PC
            
            st.write('Attention! Le tableau interractif nécessite une connexion haut-débit sinon il fera planter le dashboard.')  
            selection = aggrid_interactive_table(df = df[(df[options]>=values[0]) & 
                                                         (df[options]<=values[1])])
            # selection = aggrid_interactive_table(df = df[(df[options[0]]>=values[0]) & 
            #                                              (df[options[0]]<=values[1])])
            print("clés de selection :", selection.keys())
            # selection.data :  données d'entrées = df 
            # selection.selected_rows :  ligne(s) sélectionnée(s) par l'utilisateur (mais ne contient pas l'index du df)
            # selection.column_state : état des colonnes de la/les ligne(s) sélectionnée(s) par l'utilisateur

            print("============selected_rows.shape =", np.shape(selection.selected_rows))
            # TO DEBUG
            if selection.selected_rows!=[] :
                # transforme l'objet selection.selected_rows en DataFrame pour en extraire l'indice.
                selected_row_index = pd.DataFrame(selection.selected_rows).rowIndex.values[0]
            
                print("selected_row_index : ", selected_row_index)
                
                # création d'une variable de session pour vérifier si l'indice change après un rechargement de la page
                if "selected_row_index" not in st.session_state:
                    st.session_state.selected_row_index=selected_row_index
                    
                if st.session_state.selected_row_index!=selected_row_index :
                    st.session_state.change_index=False
                    st.session_state.selected_row_index=selected_row_index
                    
                st.write("Index sélectionné dans le DataFrame affiché est : ",
                         selected_row_index)  # ce n'est pas l'index de df
                data = pd.DataFrame(selection.selected_rows)
                data = data[features_names].values.tolist()
    
    # Pour permettre de sélectionner un index autre que celui filtré
    if 'change_index' not in st.session_state:
        st.session_state.change_index=False
    def change_index():
        st.session_state.change_index=True

    sample_index = st.number_input("Si vous souhaitez sélectionner un client directement\
                                    par son id dans le DataFrame d'origine, \
                                   indiquez le, sinon appuyez directement sur \
                                   'Prédire'",
                                   min_value=-1, max_value=len(df),  step=1, 
                                   on_change=change_index)
    # st.write("change_index", st.session_state.change_index)
    
    if st.session_state.change_index :
        data = [list(df.iloc[sample_index].values)]

    predict_btn = st.button('Prédire')
       
    if predict_btn:
        
        st.write("Demande de prédiction en cours...")
        
        pred = None

        pred = request_prediction(FLASK_URI, data, features_names) # [0]
        
        display_result(pred)
        # Transforme la liste de pred["shap_values"] en valeur numérique
#         pred["shap_values"] = [round(float(value),3) for value in pred["shap_values"]]
        
#         st.write("La proba d'une défaillance du client est de {:.2f}"
#                  .format(pred["proba"]))
#         print("=== Les features plus importantes lors de cette prédiction sont : === {}"
#                 .format(pred["features"]))
#         print("=== Les shapley_values correspondantes sont === {} "
#                 .format(pred["shap_values"]))

#         features_importances = pd.DataFrame(pred["shap_values"], index=pred["features"])
#         features_importances["influence"]=["positive" if value>0 
#                                         else "negative" 
#                                         for value in pred["shap_values"]]
        
#         # features_importances["sorting"] = [str(i) for i in features_importances.index]
        
#         red = px.colors.qualitative.Set1[0]
#         blue = px.colors.qualitative.Set1[1]
#         color = [ [blue, red] if pred["shap_values"][0]>0 else [red, blue] ]
#         # color_discrete_sequence = [red if value<0 else blue
#         #                            for value in pred["shap_values"]]
#         # Comportement étrange: color_discrete_sequence doit avoir la dimension de 
#         #                       features_importances["influence"] mais plotly ne prend
#         #                       que les 2 premières comme si il affectait ces valeurs
#         #                       à "negative" et "positive".
#         color_discrete_sequence = [None]*10
#         color_discrete_sequence[:2] = color[0]
#         print(color_discrete_sequence)
#         fig = px.bar(features_importances,
#                      title="Importance des features sur la probabilité de défaillance",
#                      labels={"index":"Features Names", "value": "Features Importances"},
#                             #"variable": "SHAPley values"},
#                      color ="influence",
#                      color_discrete_sequence=color_discrete_sequence)

#         fig.update_layout(xaxis={'categoryorder': "array",
#                                  'categoryarray': 
#                                          [str(i) for i in features_importances.index]})
#         st.plotly_chart(fig)
    
if __name__ == '__main__':
    main()


