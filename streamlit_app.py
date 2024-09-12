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
stock_info_file = '20240423_PEA_stocks_info.csv'
dico_name_isin_file = 'dictionnaire_nom_isin.json'
dico_isin_name_file = 'dictionnaire_isin_nom.json'
dico_isin_ticker_file = 'dictionnaire_isin_ticker.json'
explainer_shap_RFC_file = 'explainer_shap_RFC.json'
explainer_shap_RFR_file = 'explainer_shap_RFR.json'
today = date.today()
today_pd_format = pd.Timestamp.today().normalize()
indices_reference = {
    "^STOXX": "Euro Stoxx 600",
    "^STOXX50E": "Euro Stoxx 50",
    "^FCHI": "CAC 40 (France)",
    "^GDAXI": "DAX (Allemagne)",
    "^IBEX": "IBEX 35 (Espagne)",
    "FTSEMIB.MI": "FTSE MIB (Italie)",
    "^AEX": "AEX (Pays-Bas)",
    "^SSMI": "SMI (Suisse)",
    "^OMX": "OMX Stockholm 30 (Suède)",
}

# Fonctions
@st.cache_resource
def read_df(file):
     df = pd.read_csv(file, index_col='isin')
     return df

@st.cache_resource
def load_model(model_file):
    return joblib.load(model_file)

def selection_model(selected_model):
        if selected_model == 'Random Forest Classifier':        
            model = rfc
        elif selected_model == 'Random Forest Regressor':
            model = rfr
        return model

@st.cache_resource
def import_stock_data(ticker):
    try:
        yf_instance = yf.download(ticker, start='2023-01-01', end='2024-09-07')
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données : {e}")
        return None
    return yf_instance

def variation_calculation(stock_data, today=today_pd_format, first_date='2024-01-01'):
    idx_first_day = stock_data.index.searchsorted(first_date)
    open = stock_data['Open'].iloc[idx_first_day]
    close = stock_data['Close'].asof(today)
    variation = (close - open) / close
    return variation

@st.cache_resource
def shap_object_reconstruction(shap_values_dict):
    shap_values = shap.Explanation(
        values=np.array(shap_values_dict['values']),
        base_values=np.array(shap_values_dict['base_values']),
        data=np.array(shap_values_dict['data']),
        feature_names=shap_values_dict['feature_names']
        )
    return shap_values

@st.cache_resource
def gain_calculation(tickers):
    gain = []
    tickers_downloaded = []
    warning = []

    for ticker in tickers:
        try:
            data = import_stock_data(ticker)
            gain.append(variation_calculation(data))
            tickers_downloaded.append(ticker)
        except Exception as e:
            st.warning(f"Warning : no data for {ticker} ({e})")
            warning.append(ticker)
    return gain, tickers_downloaded, warning

def strategie_1(gains):
    """
    Calcul de la performance pour un portefeuille équipondéré
    """
    try:
        perf = sum(gains) / len(gains)
    except Exception as e:
        st.error(f"Erreur : {e}")
        return None
    return perf

def strategie_2(gains, tickers, portfolio):
    """
    Calcul de la performance vpour un portefeuille pondéré par la variation de cours prédite (RFR)
    """
    try:
        weighted_gains = []
        variations = []
        for i, ticker in enumerate(tickers):
            variation = portfolio.loc[portfolio['Ticker']==ticker, 'Pred'].values[0]
            variations.append(variation)
            weighted_gains.append(gains[i] * variation)
        perf = sum(weighted_gains) / sum(variations)
    except Exception as e:
        st.error(f"Erreur : {e}")
        return None
    return perf
    
# Initialisation Python
data_2024 = read_df(data_2024_file)

stock_info = read_df(stock_info_file)

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

with open(explainer_shap_RFR_file, 'r') as f:
    shap_values_RFR_dict = json.load(f)

features = [
    'shortName',
    'longName',
    'symbol',
    'website',
    'country',
    'industry',
    'sector',
    'fullTimeEmployees',
    'marketCap',
    'currency',
    'exchange'
    ]

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

    isin = dico_name_isin[action]['isin']
    
    st.table(stock_info.loc[isin, features])
    st.write(stock_info.loc[isin, 'longBusinessSummary'])

    st.subheader(f'Variation du cours de l\'action {action} en 2024')

    ticker = dico_isin_ticker[isin]
    cours = import_stock_data(ticker) 

    try:
        return_2024 = variation_calculation(cours)
        st.write(f'{pd.Series(cours.index).dt.date.iloc[-1]} : {ticker}')
        st.markdown(f'Le cours de l\'action {action} a varié de <span style="color:blue; font-weight:bold;">{round(return_2024 * 100, 2)}%</span> \
            depuis le 1er janvier 2024.', unsafe_allow_html=True)
  
        fig = px.line(
            cours.reset_index(), x='Date', y=['Open', 'Close'],
            title=f'Cours de l\'action {action}'
            )
        fig.add_vline(x='2024-01-01', line_width=3, line_dash="dash", line_color="black")
    
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Erreur : {e}")

    st.subheader(f'Prédiction pour l\'année 2024')

    choix = ['Random Forest Classifier', 'Random Forest Regressor']
    option = st.selectbox('Choix du modèle :', choix, key ='model_choice')
    st.write(f'Le modèle choisi est {option}.')

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

        st.subheader('Interprétabilité SHAP de la prédiction')

        row_number = data_2024.index.get_loc(isin)
        st.markdown('**Force Plot :**')
        st_shap(shap.plots.force(explainer[..., 1][row_number]), height=150, width=1000)
        st.markdown('**Waterfall :**')
        st_shap(shap.plots.waterfall(explainer[..., 1][row_number]), height=500, width=1000)


    elif option == 'Random Forest Regressor':
        explainer = shap_object_reconstruction(shap_values_RFR_dict)
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

        row_number = data_2024.index.get_loc(isin)
        st.markdown('**Force Plot :**')
        st_shap(shap.plots.force(explainer[row_number]), height=150, width=1000)
        st.markdown('**Waterfall :**')
        st_shap(shap.plots.waterfall(explainer[row_number]), height=500, width=1000)

    st.text("")
    st.text("")
    st.markdown(
        """
        Les variables sont :
        - **Return_n** : Variation du cours boursier en 2023.
        - **country** : Pays.
        - **new_industry** : Classification construite d'après le secteur d'activité et l'activité de la société.
        - **exchange** : Place de cotation de la société.
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

    st.subheader("Sélection du modèle")
    choix_2 = ['Random Forest Classifier', 'Random Forest Regressor', 'Mixte']
    option_2 = st.selectbox('Choix du modèle :', choix_2, key ='model_choice')
    st.write(f'Le modèle choisi est {option_2}.')

    if option_2 != "Mixte":
        modele = selection_model(option_2)

    init_nb_action = 25

    nb_action = st.slider(
        'Nombre d\'actions :',
        min_value=1,
        max_value=100,
        value=init_nb_action
        )
    
    if option_2 == 'Random Forest Classifier':
        probs = modele.predict_proba(data_2024)
        
        df_predicted = pd.concat(
            [
                data_2024,
                pd.DataFrame(probs[:,1], index=data_2024.index, columns=['Probabilité'])
                ],
                axis=1
                )
        

        df_predicted['Nom'] = df_predicted.index.map(dico_isin_name)
        df_predicted['Ticker'] = df_predicted.index.map(dico_isin_ticker)
    
        portfolio = df_predicted.sort_values(by='Probabilité', ascending=False)
        portfolio = portfolio.set_index('Nom')
        portfolio = portfolio.head(nb_action)
        st.table(portfolio[['Ticker','Probabilité']].head(nb_action))

    elif option_2 == 'Random Forest Regressor':
        y_pred = modele.predict(data_2024)  
    
        df_predicted = pd.concat(
            [
                data_2024,
                pd.DataFrame(y_pred, index=data_2024.index, columns=['Variation'])
                ],
                axis=1
                )
        
        df_predicted['Nom'] = df_predicted.index.map(dico_isin_name)
        df_predicted['Ticker'] = df_predicted.index.map(dico_isin_ticker)
    
        portfolio = df_predicted.sort_values(by='Variation', ascending=False)
        portfolio = portfolio.set_index('Nom')
        portfolio = portfolio.head(nb_action)
        st.table(portfolio[['Ticker','Variation']].head(nb_action))
    
    elif option_2 == 'Mixte':
        probs = rfc.predict_proba(data_2024)
        y_pred = rfr.predict(data_2024)

        combine = probs[:,1] * y_pred

        df_predicted = pd.concat(
            [
                data_2024,
                pd.DataFrame(combine, index=data_2024.index, columns=['Combinaison'])
                ],
                axis=1
                )
        
        df_predicted['Nom'] = df_predicted.index.map(dico_isin_name)
        df_predicted['Ticker'] = df_predicted.index.map(dico_isin_ticker)
    
        portfolio = df_predicted.sort_values(by='Combinaison', ascending=False)
        portfolio = portfolio.set_index('Nom')
        portfolio = portfolio.head(nb_action)
        st.table(portfolio[['Ticker','Combinaison']].head(nb_action))
    

    selected_rows = []

    with st.expander("Sélection d'un panier d'actions"):
        st.write("Cochez les actions et sélectionnez")

        for i, nom in enumerate(portfolio.index.values):
            if st.checkbox(
                f"{portfolio.loc[portfolio.index == nom].index.values[0]}",
                key=i,
                value=nom in selected_rows
                ):
                selected_rows.append(nom)

        st.write("Nombre d'actions retenues :", len(selected_rows))
        
    if selected_rows:
        selected_portfolio = portfolio[portfolio.index.isin(selected_rows)]
    else:
        selected_portfolio = portfolio

    gains, tickers, warnings = gain_calculation(selected_portfolio['Ticker'].to_list())

    st.subheader("Sélection de la stratégie")
    strategies = [
        'Portefeuille équipondéré',
        'Portefeuille pondéré'
        ]
    strategie = st.selectbox('Choix de la stratégie :', strategies, key ='strategie_choice')
    st.write(f'La stratégie choisie est {strategie}.')

    def generate_display_text_2(performance):
        """
        Génère le texte à afficher en fonction de la performance
        """
        if performance >= 0:
            color = 'green'
        else:
            color = 'red'
            
        return f"<span style='color:{color}; font-weight:bold;'>{round(performance * 100, 2)}%</span>"            
                    
    if strategie == 'Portefeuille équipondéré':
        performance_pe = strategie_1(gains)
        perf_display_pe = generate_display_text_2(performance_pe)
        st.markdown(f"Depuis le 1er janvier 2024, la performance avec un portefeuille équipondéré \
                    de {len(gains)} actions est de " + perf_display_pe +".", unsafe_allow_html=True)

    elif strategie == 'Portefeuille pondéré':
        if option_2 == 'Random Forest Classifier':         
            performance_pp = strategie_2(gains, tickers, selected_portfolio.rename(columns={'Probabilité' : 'Pred'}))
        elif option_2 == 'Random Forest Regressor':
            performance_pp = strategie_2(gains, tickers, selected_portfolio.rename(columns={'Variation' : 'Pred'}))
        elif option_2 == 'Mixte':
            performance_pp = strategie_2(gains, tickers, selected_portfolio.rename(columns={'Combinaison' : 'Pred'}))       
        perf_display_pp = generate_display_text_2(performance_pp)
        st.markdown(f"Depuis le 1er janvier 2024, la performance avec un portefeuille de {len(gains)} actions pondéré \
                    est de " + perf_display_pp +".", unsafe_allow_html=True)

    st.text("")
    st.text("")
    st.markdown('__Indices de référence depuis le 1er janvier 2024 :__')
    
    references, ref_tickers, ref_warnings = gain_calculation([k for k in indices_reference.keys()])

    for i, indice in enumerate(ref_tickers):
        st.write(f"{indices_reference[indice]} : {round(references[i] * 100,2)}%")


    


