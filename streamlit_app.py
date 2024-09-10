import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import date
import plotly.express as px
import json
import shap
from streamlit_shap import st_shap

# Définition des variables
rfc_file = '20240909_Classifier_RFC.joblib'
rfr_file = '20240909_Regressor_RFR.joblib'
data_2024_file = 'preprocessed_data_2024.csv'
dico_name_isin_file = 'dictionnaire_nom_isin.json'
dico_isin_name_file = 'dictionnaire_isin_nom.json'
dico_isin_ticker_file = 'dictionnaire_isin_ticker.json'
explainer_shap_RFC_file = 'explainer_shap_RFC.json'
today = date.today()
today_pd_format = pd.Timestamp.today().normalize()

# Fonctions
@st.cache_data
def read_df():
     data_2024 = pd.read_csv(data_2024_file, index_col='isin')
     return data_2024

@st.cache_data
def load_model(model_file):
    return joblib.load(model_file)

def selection_model(selected_model):
        if selected_model == 'Random Forest Classifier':        
            model = rfc
        elif selected_model == 'Random Forest Regressor':
            model = rfr
        return model

def prediction(modele, isin):
     return modele.predict(data_2024[isin])

@st.cache_data
def import_stock_data(ticker):
    try:
        yf_instance = yf.download(ticker, start='2023-01-01', end='2024-09-07')
    except Exception as e:
        print(f"Erreur lors du téléchargement des données : {e}")
        return None
    return yf_instance

def return_calculation(stock_data, today=today_pd_format, first_date='2024-01-01'):
    idx_first_day = stock_data.index.searchsorted(first_date)
    open = stock_data['Open'].iloc[idx_first_day]
    close = stock_data['Close'].asof(today)
    return (close - open) / close

@st.cache_resource
def shap_object_reconstruction(shap_values_dict):
    shap_values = shap.Explanation(
        values=np.array(shap_values_dict['values']),
        base_values=np.array(shap_values_dict['base_values']),
        data=np.array(shap_values_dict['data']),
        feature_names=shap_values_dict['feature_names']
        )
    return shap_values

# Initialisation Python
data_2024 = read_df()

rfc = load_model(rfc_file)
rfr = load_model(rfr_file)

with open(dico_name_isin_file, 'r') as f:
    dico_name_isin = json.load(f)

with open(dico_isin_name_file, 'r') as f:
    dico_isin_name = json.load(f)

with open(dico_isin_ticker_file, 'r') as f:
    dico_isin_ticker = json.load(f)

with open(explainer_shap_RFC_file, 'r') as f:
    shap_values_RFC_dict = json.load(f)

# Streamlit
st.title("Online Portfolio Allocation")
st.sidebar.title("Sommaire")
pages=[
    "Introduction",
    "Données",
    "Visualisation",
    "Machine Learning",
    "Prédiction",
    "Stratégies OPA",
    ]
page=st.sidebar.radio("Aller vers", pages)

# Introduction
if page == pages[0]:
    st.header('Introduction')
    st.markdown('L\'objectif de ce projet est de créer un modèle d\'allocation de portefeuille qui adapte sa stratégie “online”.')

# Données
if page == pages[1]:
    st.header('Données')

# Visualisation
if page == pages[2]:
    st.header('Visualisation')

# Machine Learning
if page == pages[3]:
    st.header("Machine Learning")

# Prédiction de la variation du cours d'une action
if page == pages[4]:
    st.header("Prédiction de la variation du cours d'une action")

    actions = [dico_isin_name[isin] for isin in data_2024.index.values]

    action = st.selectbox('Choix de l\'action :', actions, key ='stock_choice')
    st.write(f'L\'action choisi est {action}.')

    st.subheader(f'Variation du cours de l\'action {action} en 2024')

    isin = dico_name_isin[action]['isin']
    ticker = dico_isin_ticker[isin]
    st.write(ticker)
    cours = import_stock_data(ticker)
    if cours is None:
        st.error(f'Erreur : Le cours de l\'action {action} n\'a pas pu être téléchargé sur Yahoo Finance')
    else:
        return_2024 = return_calculation(cours)

        st.write(f'{pd.Series(cours.index).dt.date.iloc[-1]}')
        st.write(f'Le cours de l\'action {action} a varié de {round(return_2024 * 100, 2)}% \
                 depuis le 1er janvier 2024')
        
        fig = px.line(
            cours.reset_index(), x='Date', y=['Open', 'Close'],
            title=f'Cours de l\'action {action}'
            )
        fig.add_vline(x='2024-01-01', line_width=3, line_dash="dash", line_color="black")
        st.plotly_chart(fig)

    st.subheader(f'Prédiction pour l\'année 2024')

    choix = ['Random Forest Classifier', 'Random Forest Regressor']
    option = st.selectbox('Choix du modèle :', choix, key ='model_choice')
    st.write('Le modèle choisi est ', option)

    modele = selection_model(option)

    if option == 'Random Forest Classifier':
        explainer = shap_object_reconstruction(shap_values_RFC_dict)
        pred = modele.predict(data_2024[data_2024.index == isin])[0]
        prob = modele.predict_proba(data_2024[data_2024.index == isin])[0,1]        

        def generate_display_text(action, pred):
            """
            Génère le texte à afficher en fonction de la prédiction.
            """
            if pred >= 0:
                color = 'green'
            else:
                color = 'red'

            variation = {
                0 : "négative",
                1 : "positive"}
            
            return f"La variation du cours de l\'action {action} est prédite \
                <span style='color:{color}; font-weight:bold;'>{variation[pred]}</span> \
                avec une probabilité de \
                <span style='color:{color}; font-weight:bold;'>{round(prob * 100, 1)}%</span>."
        
                
        to_display = generate_display_text(action, pred)
        st.write('La prédiction est réalisée sur la base des données de l\'année 2023.')
        st.markdown(to_display, unsafe_allow_html=True)

    elif option == 'Random Forest Regressor':
        # explainer = shap_object_reconstruction(shap_values_RFR_dict)
        pred = modele.predict(data_2024[data_2024.index == isin])[0]
        
        def generate_display_text(action, pred):
            """
            Génère le texte à afficher en fonction de la prédiction.
            """
            if pred >= 0:
                color = 'green'
            else:
                color = 'red'
            return f"La variation de cours prédite pour l'action {action} est \
                <span style='color:{color}; font-weight:bold;'>{round(pred * 100, 1)}%</span>."
        
        to_display = generate_display_text(action, pred)
        
        st.write('La prédiction est réalisée sur la base des données de l\'année 2023.')
        st.markdown(to_display, unsafe_allow_html=True)

    st.subheader('Interprétabilité SHAP de la prédiction')

    # row_number = data_2024.index.get_loc(isin)
    # st.markdown('**Force Plot :**')
    # st_shap(shap.plots.force(explainer[..., 1][row_number]), height=150, width=1000)
    # st.markdown('**Waterfall :**')
    # st_shap(shap.plots.waterfall(explainer[..., 1][row_number]), height=500, width=1000)

    st.write('\n\n\n\n')
    st.markdown(
        """
        Les variables sont :
        - **Return_n** : Variation du cours boursier en 2023.
        - **country** : Pays d'origine.
        - **new_industry** : Classification construite d'après le secteur d'activité et l'activité de la société.
        - **exchange** : Lieu de cotation de la société.
        - **fte_category** : TPE, PME, ETI, GE. 
        - **cap_category** : Micro-Cap, Small-Cap, Mid-Cap, Large-Cap
        - **businessClass** : Classification construite d'après la description de la société.
        - **Total Revenue** : Chiffre d'affaires total généré par une entreprise à partir de ses activités principales.
        - **Net Income** : Bénéfice net après déduction de toutes les charges, dépenses, impôts et intérêts.
        - **EBITDA** : Bénéfices avant intérêts, impôts, dépréciation et amortissement, indiquant la rentabilité opérationnelle.
        - **Basic EPS** : Bénéfice par action.
        - **Operating Cash Flow** : Flux de trésorerie généré par les activités opérationnelles.
        - **Free Cash Flow** : Flux de trésorerie disponible après les dépenses en capital et les investissements nécessaires.
        - **Total Assets** : Somme de tous les actifs détenus par une entreprise.
        - **Long Term Debt** : Dette financière à long terme.
        - **Total Liabilities Net Minority Interest** : Total des passifs, y compris la dette et autres obligations.
        - **Stockholders Equity** : Valeur nette des capitaux propres après remboursement de toutes les dettes.
        
        **Annual Variation** : Variation de la variable en 2023.
        """
        )


# Stratégies OPA
if page == pages[5]:
    st.header("Stratégies Online Portfolio Allocation")


