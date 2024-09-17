import streamlit as st
from st_social_media_links import SocialMediaIcons
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import date
import plotly.express as px
import json
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import seaborn as sns

# Définition des variables
DataScientest = """https://datascientest.com/"""
rfc_file = '20240909_Classifier_RFC.joblib'
rfr_file = '20240909_Regressor_RFR.joblib'
stock_data_file = '20240719_df_v2.csv'
data_2024_file = 'preprocessed_data_2024.csv'
stock_info_file = '20240423_PEA_stocks_info.csv'
reference_finale_file = 'Reference_finale.csv'
dico_name_isin_file = 'dictionnaire_nom_isin.json'
dico_isin_name_file = 'dictionnaire_isin_nom.json'
dico_isin_ticker_file = 'dictionnaire_isin_ticker.json'
dico_exchange_file = 'dictionnaire_exchange.json'
lemmatizer_pic_file = 'lemmatization.jpg'
vectorizer_pic_file = 'vectorization.jpg'
CAH_dendogramme_file = 'CAH_dendogramme.jpg'
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

disclaimer = """
    L'application Online Portfolio Allocation est conçue à des fins pédagogiques uniquement. Les informations et 
    stratégies d'allocation de portefeuille boursier présentées sur cette plateforme ne constituent en aucun cas 
    des conseils en investissements. Nous ne garantissons pas l'exactitude, l'exhaustivité ou la pertinence des 
    informations fournies.  
    Les utilisateurs sont invités à consulter un conseiller financier professionnel avant de prendre toute décision d'investissement.  
    L'utilisation de cette application se fait à vos propres risques, et Online Portfolio Allocation décline toute 
    responsabilité en cas de pertes ou de dommages résultant de l'utilisation des informations contenues sur cette plateforme.
"""

# Fonctions
def disclaimer_display(disclaimer=disclaimer):
    for i in range (10):
        st.write("")
    st.info(disclaimer, icon="ℹ️")

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
        yf_instance = yf.download(ticker, start='2023-01-01', end='2024-12-31')
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

stock_data = read_df(stock_data_file)

reference_finale = read_df(reference_finale_file)

rfc = load_model(rfc_file)
rfr = load_model(rfr_file)

with open(dico_name_isin_file, 'r') as f:
    dico_name_isin = json.load(f)

with open(dico_isin_name_file, 'r') as f:
    dico_isin_name = json.load(f)

with open(dico_isin_ticker_file, 'r') as f:
    dico_isin_ticker = json.load(f)

dico_ticker_isin = {v : k for k , v in dico_isin_ticker.items()}    

with open(dico_exchange_file, 'r') as f:
    dico_exchange = json.load(f)

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
    "Analyse des Données",
    "Machine Learning",
    "Prédiction",
    "Stratégies OPA",
    ]
page=st.sidebar.radio("Aller vers", pages)

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.markdown("**Guillaume BARAIS**")

social_media_links = [
        "https://www.linkedin.com/in/guillaume-barais",
        "https://github.com/guillaumebarais/",]

social_media_icons = SocialMediaIcons(social_media_links) 

social_media_icons.render(sidebar=True, justify_content="left")

# Introduction
if page == pages[0]:
    st.header('Introduction')
    st.subheader('Projet')
    st.markdown("""Le projet Online Portfolio Allocation est un projet de Data Science réalisé
                dans le cadre de la [formation Data Scientist](https://datascientest.com/formation-data-scientist)
                 de l'école [DataScientest](https://datascientest.com/).
                """)
    
    st.markdown("""Un projet d'**Online Portfolio Allocation** (OPA) consiste à utiliser des algorithmes et des techniques de machine learning pour gérer et optimiser un portefeuille d'investissements en ligne.""")
    st.markdown("""
                Quelques points clés :
                * **Objectif** : Maximiser les rendements tout en minimisant les risques en répartissant les investissements de manière optimale entre différentes actions.
                * **Données** : Utilisation de données financières historiques et en temps réel pour prendre des décisions d'investissement informées.
                * **Algorithmes** : Implémentation d'algorithmes de Machine Learning pour prévoir les rendements et les probabilités.
                * **Personnalisation** : Adaptation des stratégies d'investissement aux préférences et aux objectifs spécifiques de chaque investisseur.
                * **Automatisation** : Automatisation des décisions d'achat et de vente pour réagir rapidement aux changements du marché.
    """)
    st.subheader('Cadrage du projet')
    st.markdown("""
                * **Périmètre** : Actions européennes éligibles au PEA.
                * **Objectif** : Créer plusieurs stratégies d'allocation personnalisable pour battre les principaux indices européens esur l'année 2024.
                * **Méthodologie** :
                    * Utilisation des donnnées de profil de l'entreprise, des données financières, des données comptables et des données boursières
                    * Analyse des données, data visualisation et pré-processing.
                    * Entraînement de modèles de machine learning de classification et de régression.
                    * Interprétabilité des résultats.
                    * Définition de stratégies d'investissement.
                            
    """)

    disclaimer_display()
             
# Données
if page == pages[1]:
    st.header('Données')
    st.subheader('Liste des actions éligibles au PEA')
    st.write("Sont éligibles au PEA, les actions de sociétés qui ont leur siège dans l'Union Européenne ou dans un État de l'Espace économique européen (EEE) : Europe des 27 + Islande, Norvège et Liechtenstein.")
    st.markdown("""
                La liste des sociétés a été construite à partir de plusieurs sources :
                * [Euronext](https://live.euronext.com/fr/products/equities/list), société qui gère plusieurs marchés boursiers européens répartis sur 6 pays : Belgique, France, Irlande, Norvège, Portugal et Pays-Bas,  
                * [STOXX Europe 600](https://www.stoxx.com/selection-lists), indice boursier composé de 600 des principales capitalisations boursières européennes,
                * [XETRA](https://www.deutsche-boerse-cash-market.com/dbcm-en/instruments-statistics/statistics/listes-companies) opéré est Deutsche Börse en Allemagne,
                * [Nasdaq Nordic](https://www.nasdaqomxnordic.com/digitalAssets/111/111951_nordic-list-2024-02-29.xlsx) qui comprend les marchés suédois, finlandais, danois, islandais et des 3 pays baltes,
                * [Bourse de Varsovie](https://www.gpw.pl/gpw-statistics#5) plus grande bourse d'Europe centrale et orientale,
                * [Site ABC Bourse](https://www.abcbourse.com/download/libelles) qui publie des données de plusieurs marchés européens comme les marchés allemand, belge, espagnol, hollandais, italien et portugais.  
                """)

    st.write("5132 sociétés sont initialement identifiées par leur ISIN (Intenational Securities Identification Number).")
    
    st.subheader('Collecte des données')

    st.write("La librairie Python [yfinance](https://github.com/ranaroussi/yfinance) a été utilisée pour accéder aux données du site [Yahoo Finance](https://finance.yahoo.com/).")
    st.write("Les données, comprenant 486 variables et couvrant 4657 sociétés sur la période de 2020 à 2024, occupent un espace de 1,75 Go.")
    st.write("__Exemple de données :__")
    
    all_actions = sorted([dico_isin_name[isin] for isin in stock_info.index.values])
    selected_action = st.selectbox(
        'Choix de l\'action',
        all_actions,
        key ='all_stock_choice',
        index=None,
        placeholder="Sélectionner une action...")
    st.write('L\'action choisie est ', selected_action, '.')
    
    if selected_action is not None:
        isin = dico_name_isin[selected_action]['isin']

        with st.echo():    
            ticker = dico_isin_ticker[isin]
            data = yf.Ticker(ticker)
            data_info = pd.DataFrame(data.info).T.rename(columns={0 : isin})
            data_balance_sheet = pd.DataFrame(data.balance_sheet)
            data_cash_flow = pd.DataFrame(data.cash_flow)
            data_financials = pd.DataFrame(data.financials)
            data_cours = yf.download(ticker).sort_values(by='Date', ascending=False)
            
            st.markdown("**Informations générales :**")
            st.write("Nombre de variables :", data_info.shape[0])
            st.dataframe(data_info.loc[:, isin])
            
            st.markdown("**Bilans annuels des bilans comptables :**")
            st.write("Nombre de variables :", data_balance_sheet.shape[0])
            st.dataframe(data_balance_sheet)
            
            st.markdown("**Bilans annuels des flux financiers :**")
            st.write("Nombre de variables :", data_cash_flow.shape[0])
            st.dataframe(data_cash_flow)
            
            st.markdown("**Bilans annuels des comptes de résultat :**")
            st.write("Nombre de variables :", data_financials.shape[0])
            st.dataframe(data_financials)
            
            st.markdown("**Cours de l'action :**")
            st.write("Nombre de variables :", data_cours.shape[1])
            st.dataframe(data_cours)
            fig = px.line(data_cours.reset_index(), x='Date', y=['Open', 'Close'], title=f'Cours d\'ouverture et de clôture')
            st.plotly_chart(fig)

    disclaimer_display()

# Analyse des Données
if page == pages[2]:
    st.header('Analyse des Données')
    st.subheader('Informations générales')
    st.write("""
             9 variables sont retenues sur les 146 variables obtenus par la méthode .info :
             * **isin** (chaîne de caractères) : Code du titre financier,
             * **name** (chaîne de caractères) : Nom de la société,
             * **country** (catégories) : Pays de la société,
             * **exchange** (catégories) : Place de cotation,
             * **sector** (catégories) : Secteur économique,
             * **industry** (catégories) : Catégorie industrielle,
             * **longBusinessSummary** (chaîne de caractères) : Description détaillée de l'activité,
             * **fullTimeEmployees** (entier) : Nombre total d'employés,
             * **marketCap (entier)** : Capitalisation boursière totale.
             """)
    
    st.subheader('Données comptables et financières')
    st.write("""
             10 variables sont retenues sur les 332 variables obtenues par les méthodes .balance_sheet, .cash_flow et .financials :
            * **Total Revenue (float)** : Chiffre d'affaires total généré par une entreprise à partir de ses activités principales.  
            * **Net Income (float)** : Bénéfice net après déduction de toutes les charges, dépenses, impôts et intérêts.  
            * **EBITDA (float)** : Bénéfices avant intérêts, impôts, dépréciation et amortissement, indiquant la rentabilité opérationnelle.
            * **Basic EPS (float)** :Bénéfice par action.  
            * **Operating Cash Flow (float)** : Flux de trésorerie généré par les activités opérationnelles.  
            * **Free Cash Flow (float)** : Flux de trésorerie disponible après les dépenses en capital et les investissements nécessaires.
            * **Total Assets (float)** : Somme de tous les actifs détenus par une entreprise.
            * **Long Term Debt (float)** : Dette financière à long terme.
            * **Total Liabilities Net Minority Interest (float)** : Total des passifs, y compris la dette et autres obligations.
            * **Stockholders Equity (float)** : Valeur nette des capitaux propres après remboursement de toutes les dettes.
             """)                

    st.subheader('Données boursières')
    st.write("""
            2 variables sont créés à partir du cours boursier :  
            * **Return_N** (float) : Variation en pourcentage entre le cours d'ouverture du premier jour suivant la clôture des résultats de l'année N-1  
            et le cours de clôture du dernier jour de cotation précédant la clôture des résultats de l'année N.  
            * **Return_N+1** (float) : **Variable cible**, variation en pourcentage entre le cours d'ouverture du premier jour suivant la clôture des résultats de l'année N  
            et le cours de clôture du dernier jour de cotation précédant la clôture des résultats de l'année N+1.
            """)

    st.subheader("Sélection de la variable à visualiser")
    variable_infos = [
        'country', 
        'exchange', 
        'industry et sector', 
        'longBusinessSummary', 
        'fullTimeEmployees',
        'marketCap',
        'Total Revenue',
        'Net Income',
        'EBITDA',
        'Basic EPS',
        'Operating Cash Flow',
        'Free Cash Flow',
        'Total Assets',
        'Long Term Debt',
        'Total Liabilities Net Minority Interest',
        'Stockholders Equity',
        'Return'
        ]

    choix_info = st.selectbox(
        'Choix d\'une variable :',
         variable_infos, 
         key ='info_choice',
        index=None,
        placeholder="Sélectionner une variable...")
    st.write('La variable choisie est ', choix_info, '.')

    if choix_info == 'country':

        st.write("#### Pays :")
        
        df_country = stock_info['country'].value_counts().reset_index()
        df_country.columns = ['Pays', 'Actions']
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_country, x='Actions', y='Pays', hue='Pays', orient='h')
        plt.xticks(ticks=range(0, df_country['Actions'].max()+1, 50))
        plt.title("Répartition par pays")
        st.pyplot(plt)

        st.write("""
                Traitement des valeurs manquantes : à partir des deux premières lettres des codes ISIN.
                Par exemple, **FR**0000131906 : France, **PL**PKN0000018 : Pologne
                """)
        
        st.write("Les pays non éligibles ont été retirés du dataset.")
        
        st.write("Nombre de modalités : ", df_country['Pays'].nunique())
        
    if choix_info == 'exchange':

        st.write("#### Place de cotation :")

        df_exchange = stock_info['exchange'].map(dico_exchange).value_counts().reset_index()
        df_exchange.columns = ['Place de cotation', 'Actions']

        plt.figure(figsize=(10,6))
        sns.barplot(data=df_exchange.iloc[:20,:], x='Actions', y='Place de cotation', hue='Place de cotation', orient='h', gap=0.2)
        plt.xticks(ticks=range(0, df_exchange['Actions'].max()+1, 50))
        plt.title("Les 20 principales places de cotation")
        st.pyplot(plt)
        
        st.write("""
                Traitement des valeurs manquantes : Pas de valeurs manquantes
              """)
        
        st.write("Nombre de modalités : ", df_exchange['Place de cotation'].nunique())

        st.warning("""Les places boursières non éligibles au PEA non pas été éliminées du dataset. Par exemple : Londres et New-York.
                   """)

    if choix_info == 'industry et sector':
    
        # Secteur économique
        st.write("#### Secteur énonomique :")

        df_sector = stock_info['sector'].value_counts().reset_index()
        df_sector.columns = ['Secteur économique', 'Actions']
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_sector, x='Actions', y='Secteur économique', hue='Secteur économique', orient='h')
        plt.xticks(ticks=range(0, df_sector['Actions'].max()+1, 50))
        plt.title("Répartition par secteur économique")
        st.pyplot(plt)

        st.write("""
                Traitement des valeurs manquantes : Création d'une catégorie 'Not Defined or Others'
                """)

        st.write("""
                Encodage : Catégories avec LabelEncoder()
                """)
        
        st.write("Nombre de modalités : ", df_sector['Secteur économique'].nunique())
        
        #  Catégorie industrielle
        st.write("#### Catégorie industrielle :")
            
        df_industry = stock_info['industry'].value_counts().reset_index()
        df_industry.columns = ['Catégorie industrielle', 'Actions']
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_industry.head(20), x='Actions', y='Catégorie industrielle', hue='Catégorie industrielle', orient='h')
        plt.xticks(ticks=range(0, df_industry['Actions'].max()+1, 50))
        plt.title("Les 20 premières catégories industrielles")
        st.pyplot(plt)

        st.write("""
                Traitement des valeurs manquantes : Création d'une catégorie 'Not Defined or Others'
                """)
         
        st.write("Nombre de modalités : ", df_industry['Catégorie industrielle'].nunique())

        # Catégorie new_industry
        st.write("#### Création d'une nouvelle variable 'new_industry'")
        st.write("""
                La catégorie industrielle est une subdivision des secteurs économiques.
                Les deux variables sont corrélées.
                """)

        st.write("""
                Création d'une nouvelle variable 'new_industry' proposant plus de modalités que la variable Secteur économique 
                en combinant les modalités Catégorie industrielle par Secteur industrielle.
                """)

        df_new_industry = stock_data.copy()
        df_new_industry = df_new_industry[~df_new_industry.index.duplicated(keep='first')]
        df_new_industry = df_new_industry.rename(columns={'new_industry' : 'New Industry'})
        plt.figure(figsize=(10,6))
        palette = sns.color_palette("hls", len(df_new_industry['New Industry'].unique()))
        sns.countplot(data=df_new_industry, x='New Industry', hue='New Industry', palette=palette, legend=False)
        plt.title("Variable New Industry")
        st.pyplot(plt)

        st.write("Nombre de modalités : ", df_new_industry['New Industry'].nunique())

    if choix_info == 'longBusinessSummary':
        st.write("#### Activité de l'entreprise :")
        
        st.write("La variable longBusinessSummary décrivant l'activité de la société contient de nombreuses informations.")

        st.text("Exemples de description :")
        st.dataframe(stock_info[['longName','longBusinessSummary']].sample(5))
            
        st.write("""
                Une façon d'exploiter ces informations est de créer une nouvelle variable catégorielle en regroupant les 
                sociétes dont les descriptions sont les plus proches.
                """)

        st.write("""
                Méthodologie :  
                * Tokenisation (_word_tokenize()_)  
                    * Suppression stopwords, nombres et dates par regex  
                    * Suppression nom du pays en fin de description (présent dans la variable 'country')  
                * Lemmatisation (_WordNetLemmatizer()_)  
                """)

        st.image(lemmatizer_pic_file, use_column_width=True)

        st.write("""  
            * Vectorisation (_TfidfVectorizer()_)  
            * Calcul du matrice de similarité (_cosine_similarity()_)
            """)

        st.image(vectorizer_pic_file, use_column_width=True)
        
        st.write("""
                * Hierarchical Clustering (_AgglomerativeClustering()_) :  
                    * Regroupement non supervisé des sociétés par similitude permettant de créer une nouvelle feature 'businessClass'  
                    * Meilleur compromis d'après le score de silhouette et le score de Calinski-Harabasz (elbow method) : 10 clusters
                """)

        df_business = stock_data.copy()
        df_business = df_business[~df_business.index.duplicated(keep='first')]
        df_business = df_business.rename(columns={'businessClass': 'Business Class'})
        plt.figure(figsize=(10,6))
        palette = sns.color_palette("hls", len(df_business['Business Class'].unique()))
        sns.countplot(data=df_business, x='Business Class', hue='Business Class', palette=palette, legend=False)
        plt.title("Variable Business Class")
        st.pyplot(plt)
        
        st.write("""
                Traitement des valeurs manquantes : associées à une classe lors du clustering non supervisé
                """)
         
        st.write("Nombre de modalités : ", df_business['Business Class'].nunique())

        st.write("")
        st.write("")
        st.text("Exemple de classification :")

        df_business = pd.merge(
            left=df_business.reset_index(),
            right=stock_info.reset_index(),
            on='isin', how='inner').set_index('isin')

        all_actions = sorted([dico_isin_name[isin] for isin in df_business.index.values])
        
        selected_for_business = st.selectbox(
            'Choix de l\'action',
            all_actions,
            key ='all_business_class',
            index=None,
            placeholder="Sélectionner une action...")
        st.write('L\'action choisie est ', selected_for_business, '.')

        if selected_for_business is not None:
            st.dataframe(
                df_business.loc[dico_name_isin[selected_for_business]['isin'], ['longName', 'Business Class', 'longBusinessSummary']],
                use_container_width=True
                )

    if choix_info == 'fullTimeEmployees':

        st.write("#### Nombre d'employés :")

        df_employees = stock_info[['longName','fullTimeEmployees','fte_category']].copy().sort_values(by='fullTimeEmployees', ascending=False).drop_duplicates(keep='first')

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(df_employees['fullTimeEmployees'].describe(), use_container_width=True)

        with col2:
            st.write("")
            st.write("")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_employees, y='fullTimeEmployees', ax=ax)
            ax.set_yscale('log')
            ax.set_ylabel('Effectifs')
            ax.set_title("Distribution des effectifs")
            st.pyplot(fig, use_container_width=True)

        fig_2, ax_2 = plt.subplots(figsize=(10,6))
        sns.barplot(data=df_employees.head(10), x='fullTimeEmployees', y='longName', hue='longName', orient='h', ax=ax_2)
        ax_2.set_xlabel('Employés')
        ax_2.set_ylabel('')
        ax_2.set_title("Les 10 premières sociétés par effectif")
        st.pyplot(fig_2, use_container_width=True)

        st.write("")
        st.write("""
                Création d'une nouvelle variable 'fte_category' permettant de regrouper les sociétés par taille d'effectif :  
                * TPE (Très Petites Entreprises) : de 0 à 19 employés,  
                * PME (Petites et Moyennes Entreprises) : de 20 à 249 employés,  
                * ETI (Etablissement de Taille Intermédiaire) : de 250 à 5 000 employés,  
                * GE (Grandes Entreprises) : + de 5 000 employés.  
                """)

        col3, col4 = st.columns(2)

        with col3:
            fig_3, ax_3 = plt.subplots(figsize=(6,6))
            sns.boxplot(data=df_employees, x='fte_category', y='fullTimeEmployees', hue='fte_category', ax=ax_3)
            ax_3.set_yscale('log')
            ax_3.set_ylabel('Effectifs')
            ax_3.set_title("Effectif par catégorie")
            st.pyplot(fig_3, use_container_width=True)

        with col4:
            dico_fte_category = {
                "1" : "TPE",
                "2" : "PME",
                "3" : "ETI",
                "4" : "GE",
            }
            
            df_employees_final = stock_data[['fte_category']]      
            categorie_name = list(df_employees_final['fte_category'].value_counts().index)
            categorie_name_mapped = [dico_fte_category[str(int(category))] for category in categorie_name]
            categorie_value = list(df_employees_final['fte_category'].value_counts())
            fig_4, ax_4 = plt.subplots(figsize=(6,6))
            plt.pie(categorie_value, labels = categorie_name_mapped, autopct='%.0f%%')
            plt.title("Répartition des entreprises par catégorie d'effectif")
            st.pyplot(fig_4, use_container_width=True)

        st.write("""
                Traitement des valeurs manquantes : KNNImputer()
                """)
        
        st.write("Nombre de modalités : ", stock_data['fte_category'].nunique())

    if choix_info == 'marketCap':

        st.write("#### Capitalisation boursière :")

        st.write("""
                Les capitalisations boursières sont exprimées dans des **devises différentes**.  
                Une **conversion en Euro** de toutes les devises a été effectuée lors d'une phase de pré-traitement. 
                """)

        st.write("")

        df_market = stock_info[['longName','marketCap','cap_category']].copy().sort_values(by='marketCap', ascending=False).drop_duplicates(keep='first')

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(df_market['marketCap'].describe(), use_container_width=True)

        with col2:
            st.write("")
            st.write("")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_market, y='marketCap', ax=ax)
            ax.set_yscale('log')
            ax.set_ylabel('Capitalisation en Euro')
            ax.set_title("Distribution de la capitalisation boursière")
            st.pyplot(fig, use_container_width=True)

        fig_2, ax_2 = plt.subplots(figsize=(10,6))
        sns.barplot(data=df_market.head(10), x=df_market.head(10)['marketCap'] * 1e-9, y='longName', hue='longName', orient='h', ax=ax_2)
        ax_2.set_xlabel('Capitalisation (milliard d\'euros)')
        ax_2.set_ylabel('')
        ax_2.set_title("Les 10 premières sociétés par capitalisation boursière")
        st.pyplot(fig_2, use_container_width=True)

        st.write("")
        st.write("""
                Création d'une nouvelle variable 'cap_category' permettant de regrouper les sociétés par capitalisation boursière :  
                * Micro-Cap (Micro-capitalisations) : <300 millions d'euros,  
                * Small-Cap (Petites capitalisations) : < 2 milliards d'euros,  
                * Mid-Cap (Moyennes capitalisations) : entre 2 et 10 milliards d'euros,  
                * Large-Cap (Grandes capitalisations) : > 10 milliards d'euros.  
                """)

        col3, col4 = st.columns(2)

        with col3:
            fig_3, ax_3 = plt.subplots(figsize=(6,6))
            sns.boxplot(data=df_market, x='cap_category', y='marketCap', hue='cap_category', ax=ax_3)
            ax_3.set_yscale('log')
            ax_3.set_ylabel('Capitalisation boursière')
            ax_3.set_title("Capitalisation boursière par catégorie")
            st.pyplot(fig_3, use_container_width=True)

        with col4:
            dico_cap_category = {
                "1" : "Micro-Cap",
                "2" : "Small-Cap",
                "3" : "Mid-Cap",
                "4" : "Large-Cap",
            }
            
            df_market_final = stock_data[['cap_category']]      
            categorie_name = list(df_market_final['cap_category'].value_counts().index)
            categorie_name_mapped = [dico_cap_category[str(int(category))] for category in categorie_name]
            categorie_value = list(df_market_final['cap_category'].value_counts())
            fig_4, ax_4 = plt.subplots(figsize=(6,6))
            plt.pie(categorie_value, labels = categorie_name_mapped, autopct='%.0f%%')
            plt.title("Répartition des entreprises par capitalisation boursière")
            st.pyplot(fig_4, use_container_width=True)

        st.write("""
                Traitement des valeurs manquantes : KNNImputer()
                """)
        
        st.write("Nombre de modalités : ", stock_data['cap_category'].nunique())

    def display_feature(feature):
        
        st.write("""
                Les données financières sont exprimées dans des **devises différentes**.  
                Une **conversion en Euro** de toutes les devises a été effectuée lors d'une phase de pré-traitement. 
                """)
        
        st.write("")

        df_feature = stock_data.loc[~(stock_data['year'] == 2024), ['year', feature, f"{feature} : Annual Variation"]].copy()
        df_feature = df_feature.sort_values(by=feature, ascending=False)

        describe = []
        for year in sorted(df_feature['year'].unique()):
            stats = df_feature.loc[df_feature['year'] == year, [feature]].describe()
            stats = stats.rename(columns={feature: f'{feature} {year}'})
            describe.append(stats)

        result = pd.concat(describe, axis=1)

        st.dataframe(result, use_container_width=True)

        if feature == 'Basic EPS':
            fig, ax = plt.subplots()
            palette = sns.color_palette("hls", len(df_feature['year'].unique()))
            sns.boxplot(data=df_feature, x='year', y=df_feature[feature], hue='year', palette=palette, ax=ax)
            ax.set_ylim(-3, 3)
            ax.set_xlabel("Année")
            ax.set_ylabel(f'{feature} (euros)')
            ax.set_title(f"Distribution de {feature}")
            st.pyplot(fig, use_container_width=True)
        elif feature == 'Long Term Debt':
            fig, ax = plt.subplots()
            palette = sns.color_palette("hls", len(df_feature['year'].unique()))
            sns.boxplot(data=df_feature, x='year', y=df_feature[feature] * 1e-6, hue='year', palette=palette, ax=ax)
            ax.set_yscale('log')
            ax.set_ylabel(f'{feature} (million d\'euros)')
            ax.set_title(f"Distribution de {feature}")
            st.pyplot(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            palette = sns.color_palette("hls", len(df_feature['year'].unique()))
            sns.boxplot(data=df_feature, x='year', y=df_feature[feature] * 1e-6, hue='year', palette=palette, ax=ax)
            ax.set_yscale('symlog')
            ax.set_ylabel(f'{feature} (million d\'euros)')
            ax.set_title(f"Distribution de {feature}")
            st.pyplot(fig, use_container_width=True)

        st.write(f"""
                Nouvelle variable : **'{feature} : Annual Variation'**  
                Cette variable est créée en calculant **la variation en pourcent (%) de {feature} par rapport à l'année précédente**.
                """)

        st.text(f"Exemple de calcul de variation annuelle")
        
        all_actions = sorted([dico_isin_name[isin] for isin in set(df_feature.index.values)])

        selected_for_feature = st.selectbox(
            'Choix de l\'action',
            all_actions,
            key ='all_feature',
            index=None,
            placeholder="Sélectionner une action...")
        st.write('L\'action choisie est ', selected_for_feature, '.')

        if selected_for_feature is not None:
            df_feature['year'] = df_feature['year'].astype('str')
            df_feature[feature] = df_feature[feature].round(0)
            df_feature[f"{feature} : Annual Variation"] = (df_feature[f"{feature} : Annual Variation"] * 100).round(1)

            df_feature = df_feature.rename(
                columns={
                    f"{feature}" : f"{feature} en euros",
                    f"{feature} : Annual Variation" : f"{feature} : Annual Variation en %"
                })

            st.dataframe(
                df_feature.loc[
                    dico_name_isin[
                        selected_for_feature]['isin'], ['year', f"{feature} en euros", f"{feature} : Annual Variation en %"]],
                use_container_width=True
                )

        st.write("")
        
        if feature == 'Basic EPS':
            st.write(f"""
                    Traitement des valeurs manquantes de 'Basic EPS' et 'Basic EPS : Annual Variation' : 
                    * Remplacement des valeurs non documentées par 0 (hypothèse pas de bénéfices)
                    """)
        elif feature == 'Long Term Debt':
            st.write(f"""
                    Traitement des valeurs manquantes de 'Long Term Debt' et 'Long Term Debt : Annual Variation' :   
                    * Remplacement des valeurs non documentées par 0 (hypothèse sociétés non endettées)
                    """)     
        else:
            st.write(f"""
                    Traitement des valeurs manquantes de '{feature}' et '{feature} : Annual Variation' :  
                    * KNNimputer()
                    """)

    if choix_info == 'Total Revenue':
        st.write("#### Chiffre d'affaire :")
        display_feature(choix_info)

    if choix_info == 'Net Income':
        st.write("#### Bénéfice net :")
        display_feature(choix_info)

    if choix_info == 'EBITDA':
        st.write("#### Bénéfices :")
        display_feature(choix_info)

    if choix_info == 'Basic EPS':
        st.write("#### Bénéfice par action :")
        display_feature(choix_info)

    if choix_info == 'Operating Cash Flow':
        st.write("#### Flux de trésorerie des activités opérationnelles :")
        display_feature(choix_info)

    if choix_info == 'Free Cash Flow':
        st.write("#### Flux de trésorerie disponible :")
        display_feature(choix_info)

    if choix_info == 'Total Assets':
        st.write("#### Somme des actifs :")
        display_feature(choix_info)

    if choix_info == 'Long Term Debt':
        st.write("#### Dette financière à long terme :")
        display_feature(choix_info)

    if choix_info == 'Total Liabilities Net Minority Interest':
        st.write("#### Total des passifs :")
        display_feature(choix_info)

    if choix_info == 'Stockholders Equity':
        st.write("#### Valeur nette des capitaux propres après remboursement des dettes :")
        display_feature(choix_info)

    if choix_info == 'Return':
        df_return = stock_data[~(stock_data['year'] == 2024)].copy()
        
        st.write("#### Return de l'année N :")

        describe = []
        for year in sorted(df_return['year'].unique()):
            stats = df_return.loc[df_return['year'] == year, ['Return_n']].describe()
            stats = stats.rename(columns={'Return_n': f'Return_n {year}'})
            describe.append(stats)

        result = pd.concat(describe, axis=1)

        st.dataframe(result, use_container_width=True)

        fig, ax = plt.subplots()    
        palette = sns.color_palette("hls", len(df_return['year'].unique()))
        sns.boxplot(data=df_return, x='year', y='Return_n', hue='year', palette=palette, showmeans=True, ax=ax)
        ax.set_ylim([-1.5, 5])
        ax.set_yticks(np.arange(-1.5, 5.5, 0.5))
        ax.set_xlabel("Année")
        ax.set_ylabel("Variation des cours par rapport à l'année précédente (%)")
        ax.set_title("Variation des cours boursiers par année")
        st.pyplot(fig, use_container_width=True)

        st.write("""Les marchés financiers sont influencés par des facteurs externes tels que des événements géopolitiques ou économiques. 
                Pour réduire l’impact de ces facteurs, **les variations sont ajustées par rapport à leur moyenne annuelle** afin d'obtenir une moyenne nulle. 
                 Cela permet de rendre les années comparables en termes de moyenne. 
                 """)
        
        st.markdown(r"""
                    Les marchés ont été très volatils en 2021 et 2022. 
                    Les performances les plus extrêmes, qu'elles soient à la hausse ou à la baisse, peuvent être considérées comme des valeurs aberrantes. 
                    Les outliers sont plafonnées à 1,5 fois l'intervalle interquartile :    
                    $$1.5 \times IQR \, (Q3 - Q1)$$.
                    """)
        
        st.write("Après traitement :")

        df_return_2 = reference_finale[~(reference_finale['year'] == 2024)].copy()

        fig_2, ax_2 = plt.subplots()    
        palette = sns.color_palette("hls", len(df_return['year'].unique()))
        sns.boxplot(data=df_return_2, x='year', y='Return_n', hue='year', palette=palette, showmeans=True, ax=ax_2)
        ax_2.set_ylim([-1.5, 5])
        ax_2.set_yticks(np.arange(-1.5, 5.5, 0.5))
        ax_2.set_xlabel("Année")
        ax_2.set_ylabel("Variation des cours par rapport à l'année précédente (%)")
        ax_2.set_title("Variation des cours boursiers par année (après pré-traitement)")
        st.pyplot(fig_2, use_container_width=True)

        st.write("#### Return de l'année N + 1 :")

        st.write("La variation de cours de l'année suivant est la variable cible à prédire.")

        df_return = stock_data[~(stock_data['year'].isin([2023, 2024]))].copy()
        df_return = df_return.dropna(subset=['Return_n+1'])

        describe = []
        for year in sorted(df_return['year'].unique()):
            stats = df_return.loc[df_return['year'] == year, ['Return_n+1']].describe()
            stats = stats.rename(columns={'Return_n+1': f'Return_n+1 {year}'})
            describe.append(stats)

        result = pd.concat(describe, axis=1)

        st.dataframe(result, use_container_width=True)

        fig_3, ax_3 = plt.subplots()    
        palette = sns.color_palette("hls", len(df_return['year'].unique()))
        sns.boxplot(data=df_return, x='year', y='Return_n+1', hue='year', palette=palette, showmeans=True, ax=ax_3)
        ax_3.set_ylim([-1.5, 5])
        ax_3.set_yticks(np.arange(-1.5, 5.5, 0.5))
        ax_3.set_xlabel("Année")
        ax_3.set_ylabel("Variation du cours (%)")
        ax_3.set_title("Variable cible : Variation des cours dans l'année")
        st.pyplot(fig_3, use_container_width=True)

        st.write("""
                 Les variations de cours au cours de l’année sont ajustées en fonction des moyennes calculées pour la 
                 variable 'Return_N'.  
                 Par exemple, la variable 'Return_N+1' pour l'année 2021 est ajustée en utilisant la moyenne de l'année 
                 2022 pour la variable 'Return_N'.
                 """)
        
        st.write("Les outliers sont traités comme pour la variable 'Return_N'.")

        st.write("Après traitement :")

        df_return_2 = reference_finale[~(reference_finale['year'].isin([2023, 2024]))].copy()

        fig_4, ax_4 = plt.subplots()    
        palette = sns.color_palette("hls", len(df_return['year'].unique()))
        sns.boxplot(data=df_return_2, x='year', y='Return_n+1', hue='year', palette=palette, showmeans=True, ax=ax_4)
        ax_4.set_ylim([-1.5, 5])
        ax_4.set_yticks(np.arange(-1.5, 5.5, 0.5))
        ax_4.set_xlabel("Année")
        ax_4.set_ylabel("Variation des cours (%)")
        ax_4.set_title("Variable cible : Variation des cours dans l'année (après pré-traitement)")
        st.pyplot(fig_4, use_container_width=True)

    disclaimer_display()

# Machine Learning
if page == pages[3]:
    st.header("Machine Learning")



# Affichage du dataframe avec encodage et standardisation

# Choix de la métrique Accuracy

# Choix des modèles Lazy predic
# Optimisation des hyper-paramètres

# Matrice de confusion et ROC_AUC

# Courbe de convergence et taux de confiance

    disclaimer_display()

# Prédiction de la variation du cours d'une action
if page == pages[4]:
    st.header("Prédiction de la variation du cours d'une action")

    actions = [dico_isin_name[isin] for isin in data_2024.index.values]

    action = st.selectbox(
        'Choix de l\'action :', 
        actions, 
        key ='stock_choice',
        index=None,
        placeholder="Sélectionner une action...")
    st.write('L\'action choisie est ', action, '.')

    if action is not None:
        
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
        option = st.selectbox(
            'Choix du modèle :', 
            choix, 
            key ='model_choice',
            index=None,
            placeholder="Sélectionner un modèle...")
        st.write('Le modèle choisi est ', option, '.')

        if option is not None:
            
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

    disclaimer_display()

# Stratégies OPA
if page == pages[5]:
    st.header("Stratégies Online Portfolio Allocation")

    st.subheader("Sélection du modèle")
    choix_2 = ['Random Forest Classifier', 'Random Forest Regressor', 'Mixte']
    option_2 = st.selectbox(
        'Choix du modèle :', 
        choix_2, 
        key ='model_choice',
        index=None,
        placeholder="Sélectionner une modèle...")
    st.write('Le modèle choisi est ', option_2, '.')

    if option_2 is not None:

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

            combine = np.sqrt(probs[:,1] * probs[:,1] + y_pred * y_pred)

            df_predicted = pd.concat(
                [
                    data_2024,
                    pd.DataFrame(combine, index=data_2024.index, columns=['Norme L2'])
                    ],
                    axis=1
                    )
            
            df_predicted['Nom'] = df_predicted.index.map(dico_isin_name)
            df_predicted['Ticker'] = df_predicted.index.map(dico_isin_ticker)
        
            portfolio = df_predicted.sort_values(by='Norme L2', ascending=False)
            portfolio = portfolio.set_index('Nom')
            portfolio = portfolio.head(nb_action)
            st.table(portfolio[['Ticker','Norme L2']].head(nb_action))

        selected_rows=[]
        
        with st.expander("Sélection d'un panier d'actions"):
            st.write("Cochez les actions...")

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
        strategie = st.selectbox(
            'Choix de la stratégie :', 
            strategies, 
            key ='strategie_choice',
            index=None,
            placeholder="Sélectionner une stratégie...")
        st.write('La stratégie choisie est ', strategie,'.')

        if strategie is not None:

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
                    performance_pp = strategie_2(gains, tickers, selected_portfolio.rename(columns={'Norme L2' : 'Pred'}))       
                perf_display_pp = generate_display_text_2(performance_pp)
                st.markdown(f"Depuis le 1er janvier 2024, la performance avec un portefeuille de {len(gains)} actions pondéré \
                            est de " + perf_display_pp +".", unsafe_allow_html=True)

    st.text("")
    st.text("")
    st.markdown('__Indices de référence depuis le 1er janvier 2024 :__')
    
    references, ref_tickers, ref_warnings = gain_calculation([k for k in indices_reference.keys()])

    for i, indice in enumerate(ref_tickers):
        st.write(f"{indices_reference[indice]} : {round(references[i] * 100,2)}%")

    st.text("")
    st.text("")
    st.markdown("*_Dummy strategy :_*")
    try:
        # Dummy Strategy
        if selected_rows:
            gains_dummy, tickers_dummy, warnings_dummy = gain_calculation(df_predicted['Ticker'].sample(len(selected_rows)).to_list())
        else:
            gains_dummy, tickers_dummy, warnings_dummy = gain_calculation(df_predicted['Ticker'].sample(len(portfolio)).to_list())
        st.write(f"_Performance de {round(strategie_1(gains_dummy) * 100,2)}% avec une liste aléatoire de {len(gains_dummy)} actions équipondérées._")
        
        st.write(f"_Liste des actions : {', '.join([dico_isin_name[dico_ticker_isin[ticker]] for ticker in tickers_dummy])}_")
    except Exception as e:
        pass

    disclaimer_display()


    


