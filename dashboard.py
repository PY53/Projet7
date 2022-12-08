import pandas as pd
import streamlit as st
import requests
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import plotly.figure_factory as ff
import plotly.express as px

# requests 2.28.1
# streamlit 1.14.0
# streamlit-aggrid 0.3.2 (installer avec Pip)
# plotly 5.11.0 (conda install -c plotly plotly=5.11.0)
# pandas 1.3.5


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
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
  
    print("============== data.shape :", np.shape(data))
    print("===================data type = ", type(data))
    data_json = {'data': data}
    print("===================data_json type = ", type(data_json))
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
    # LOAD_MODEL_URI = 'http://127.0.0.1:5000/loading_model'
    # response = requests.request(method='GET',  url=LOAD_MODEL_URI)
    # print("===================model loaded =====")
    
    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['Flask', 'Cortex', 'Ray Serve'])

    st.title('Failure risk estimation for loan delivery')

    print("Lecture de la base de données")
    df = pd.read_csv('dataset/data_for_mini_model.csv') # nrows=6)
    
    st.write("Data shape = {} ".format(df.shape))
    
    options = st.multiselect(
    'Quelles features souhaitez vous filtrer ?',
    df.columns, "CNT_CHILDREN")

    st.write('You selected:', options[0])
   
    # filter_btn = st.button('Filtrer')
    
    if options:

        # Histogram data
        x = df[options[0]].values

        # Group data together
        hist_data = [x]
        # hist_data = [x1, x2, x3]

        group_labels = ['Group 1']
        # group_labels = ['Group 1', 'Group 2', 'Group 3']

#         # Create distplot with custom bin_size
        # fig = ff.create_distplot(hist_data, group_labels, show_curve =False) # bin_size=[.1, .25, .5]   
#         # Plot!
        # st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(x=df[options[0]].values)
        st.plotly_chart(fig) # 
    
        # x_min = np.round(x.min(), 2)
        # x_max = np.round(x.max(), 2)
        x_min = float(x.min())
        x_max = float(x.max())
        
        # values = filter_slider(x_min, x_max)
        # def filter_slider(x_min, x_max):
        values = st.slider('Select a range of values',
                                x_min, x_max, (x_min, x_max)) 
        st.write('values:', values)
        
        # réduction du nombre de lignes, au-dessus de 200 MB 
        # (pandas) utilise environ 4 fois la taille du tableau en mémoire 
        number_of_rows = ((df[options[0]]>=values[0]) & (df[options[0]]<=values[1])).sum()
        st.write('Number of rows:', number_of_rows )  
        
        # proceed_btn = st.button('Proceed')
        
        selection=[]
        if number_of_rows<1000 :  # En théorie 5000 ça passe mais ça ramme sur le PC
            
            selection = aggrid_interactive_table(df = df[(df[options[0]]>=values[0]) & (df[options[0]]<=values[1])])
            st.write("clés de selection :", selection.keys())
            # selection.data :  données d'entrées = df 
            # selection.selected_rows :  ligne(s) sélectionnée(s) par l'utilisateur (mais ne contient pas l'index du df)
            # selection.column_state : état des colonnes de la/les ligne(s) sélectionnée(s) par l'utilisateur

            # TO DEBUG
            if selection.column_state :
                # transforme l'objet selection.selected_rows en DataFrame pour en extraire l'indice.
                selected_row_index = pd.DataFrame(selection.selected_rows).rowIndex.values[0]
            
                st.write("Vous avez selectionné l'index:", selected_row_index)  # ce n'est pas l'index de df
                data = selection.selected_rows[0]
                st.session_state.change_index=False
    
    # Pour permettre de sélectionner un index autre que celui filtré
    if 'change_index' not in st.session_state:
        st.session_state.change_index=False
    def data_from_index():
        st.session_state.change_index=True
    sample_index = st.number_input("Si vous souhaitez sélectionner un autre client, indiquez son index sinon appuyez directement sur 'Prédire'",
                                                  min_value=-1, max_value=len(df),  step=1, on_change=data_from_index)
    st.write("change_index", st.session_state.change_index)
    
    if st.session_state.change_index :
        data = [list(df.iloc[sample_index].values)]

    predict_btn = st.button('Prédire')
       
    if predict_btn:
        
        # data = [list(df.iloc[sample_index].values)]

        # data = [[revenu_med, age_med, nb_piece_med, nb_chambre_moy,
        #          taille_pop, occupation_moy, latitude, longitude]]
        # print("===================data = {} ".format(data))
        st.write("Data shape = {} ".format(np.shape(data)))
        # [[3.87, 28.0, 5.0, 1.0, 1425, 3.0, 35.0, -119.0]]
        
        pred = None

        # if api_choice == 'Flask':
        pred = request_prediction(FLASK_URI, data)[0]
        # elif api_choice == 'Cortex':
            # pred = request_prediction(CORTEX_URI, data)[0] * 100000
        # elif api_choice == 'Ray Serve':
            # pred = request_prediction(RAY_SERVE_URI, data)[0] * 100000
        st.write(
            "La proba d'une défaillance du client est de {:.2f}".format(pred))

if __name__ == '__main__':
    main()


