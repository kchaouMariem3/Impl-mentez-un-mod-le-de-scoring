import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import zipfile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sklearn
import lightgbm as lgb
from lightgbm import LGBMClassifier
import shap
from streamlit_shap import st_shap
from urllib.request import urlopen
import json
import requests

import pickle
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px


# Configuration de la page #

st.set_page_config(
        page_title='Score du Client',
        page_icon = "📊",
        layout="wide" )
# Titre et sous_titre du projet
st.markdown("""
            <p style="color:#772b58;text-align:center;font-size:2.8em;font-style:italic;font-weight:700;font-family:'Roboto Condensed';margin:0px;">
            Prêt à dépenser</p>
            """, 
            unsafe_allow_html=True)
st.markdown("""
            <p style="color:#d61e8b;text-align:center;font-size:1.5em;font-style:italic;font-family:'Roboto Condensed';margin:0px;">
             OpenClassrooms Projet n°7 - Data Scientist - Kchaou Mariem</p>
            """, 
            unsafe_allow_html=True)


# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('Image.png')
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")


# Lecture des fichiers #

@st.cache 
def X_test_original():
    with zipfile.ZipFile ("data.zip", "r") as fz:
         X_test_original = fz.read ("X_test_original.csv")
        
    X_test_original = pd.read_csv("X_test_original.csv")
    X_test_original = X_test_original.rename(columns=str.lower)
         
    return X_test_original

 

@st.cache 
def X_test_clean():
    X_test_clean = pd.read_csv("X_test_clean.csv")
    #st.dataframe(X_test_clean)
    return X_test_clean

@st.cache
def valeurs_shap():
    model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
    explainer = shap.TreeExplainer(model_LGBM)
    shap_values = explainer.shap_values(X_test_clean().drop(labels="sk_id_curr", axis=1))
    return shap_values


if __name__ == "__main__":

    X_test_original()
    X_test_clean()
    valeurs_shap() 

    # Titre 1
    st.markdown("""
             <h1 style="color:#3aa1a2;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                1. Quel est le score du client ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")
 
    
    # Création et affichage du sélecteur du numéro de client #
    
liste_clients = list(X_test_original()['sk_id_curr'])
col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
with col1:
        ID_client = st.selectbox("*Veuillez sélectionner l'identité numérique de votre client à l'aide du menu déroulant 👇*", 
                                (liste_clients))
        st.write("Vous avez sélectionné l'identité numérique n° :", ID_client)
with col2:
        st.write("")

        
payload = {'id_client': ID_client}
r=requests.get('http://127.0.0.1:5000/get_score_credit', params=payload)
       
 
pred_proba = json.loads(r.text)["score"]
        
             
#payload = {'id_client': ID_client}
#r=requests.get('http://127.0.0.1:5000/get_score_credit', params=payload)


score=pred_proba
#API_url =equests.get ("http://127.0.0.1:5000/get_score_credit", params=ID_client)
#result=requests.get ("http://127.0.0.1:5000/get_score_credit") 


#score= list(dict_new.value())[0]

    # Lecture du modèle de prédiction et des scores #
    
model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
y_pred_lgbm = model_LGBM.predict(X_test_clean().drop(labels="sk_id_curr", axis=1))    # Prédiction de la classe 0 ou 1
y_pred_lgbm_proba = model_LGBM.predict_proba(X_test_clean().drop(labels="sk_id_curr", axis=1)) # Prédiction du % de risque

    # Récupération du score du client
y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],
                                    X_test_clean()['sk_id_curr']], axis=1)
           
            
#score=r.text 



    

    # Récupération de la décision
y_pred_lgbm_df = pd.DataFrame(y_pred_lgbm, columns=['prediction'])
y_pred_lgbm_df = pd.concat([y_pred_lgbm_df, X_test_clean()['sk_id_curr']], axis=1)
y_pred_lgbm_df['client'] = np.where(y_pred_lgbm_df.prediction == 1, "non solvable", "solvable")
y_pred_lgbm_df['decision'] = np.where(y_pred_lgbm_df.prediction == 1, "refuser", "accorder")
solvabilite = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "client"].values
decision = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "decision"].values



# Affichage du score et du graphique de gauge sur 2 colonnes #


col1, col2 = st.columns(2)
with col2:
        st.markdown(""" <br> <br> """, unsafe_allow_html=True)
        st.write(f"Le client dont l'identité numérique est {ID_client} a obtenu le score de {score }.")
        st.write(f"Il y a donc un risque de {score:} que le client ait des difficultés de paiement.")
        st.write(f"Le client est donc considéré par 'Prêt à dépenser' comme {solvabilite[0]} \
                et décide de lui {decision[0]} le crédit. ")
    # Impression du graphique jauge
with col1:
        fig = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = float(score),
                        mode = "gauge+number+delta",
                        title = {'text': "Score du client", 'font': {'size': 24}},
                        delta = {'reference': 50, 'increasing': {'color': "#3b203e"}},
                        gauge = {'axis': {'range': [None, 100],
                                'tickwidth': 3,
                                'tickcolor': 'darkblue'},
                                'bar': {'color': 'white', 'thickness' : 0.3},
                                'bgcolor': 'white',
                                'borderwidth': 1,
                                'bordercolor': 'gray',
                                'steps': [{'range': [0, 20], 'color': '#e8af92'},
                                        {'range': [20, 40], 'color': '#db6e59'},
                                        {'range': [40, 60], 'color': '#b43058'},
                                        {'range': [60, 80], 'color': '#772b58'},
                                        {'range': [80, 100], 'color': '#3b203e'}],
                                'threshold': {'line': {'color': 'white', 'width': 8},
                                            'thickness': 0.8,
                                            'value': 50 }}))

        fig.update_layout(paper_bgcolor='white',
                        height=400, width=500, 
                        font={'color': '#772b58', 'family': 'Roboto Condensed'},
                        margin=dict(l=30, r=30, b=5, t=5))
        st.plotly_chart(fig, use_container_width=True)

    # Explication de la prédiction #
st.markdown("""
             <h1 style="color:#3aa1a2;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                2. Profil du client </h1>
                """, 
                    unsafe_allow_html=True)
st.write("")
st.write(f"Genre : **{X_test_original()['code_gender'].values[0]}**")
st.write(f"Tranche d'âge : **{X_test_original()['age_client'].values[0]}**")
st.write(f"Ancienneté de la piède d'identité : **{X_test_original()['anciennete_cni'].values[0]}**")
st.write(f"Situation familiale : **{X_test_original()['name_family_status'].values[0]}**")
st.write(f"Taille de la famille : **{X_test_original()['taille_famille'].values[0]}**")
st.write(f"Nombre d'enfants : **{X_test_original()['nbr_enfants'].values[0]}**")
st.write(f"Niveau d'éducation : **{X_test_original()['name_education_type'].values[0]}**")
st.write(f"Revenu Total Annuel : **{X_test_original()['total_revenus'].values[0]} $**")
st.write(f"Type d'emploi : **{X_test_original()['name_income_type'].values[0]}**")
st.write(f"Ancienneté dans son entreprise actuelle : **{X_test_original()['anciennete_entreprise'].values[0]}**")
st.write(f"Type d'habitation : **{X_test_original()['name_housing_type'].values[0]}**")
st.write(f"Densité de la Population de la région où vit le client : **{X_test_original()['pop_region'].values[0]}**")
st.write(f"Evaluations de *'Prêt à dépenser'* de la région où vit le client : \
                   **{X_test_original()['region_rating_client'].values[0]}**")
    # Titre 2
st.markdown("""
                <h1 style="color:#3aa1a2;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                3. Les variables globales</h1>
                """, 
                unsafe_allow_html=True)
st.write("")
explainer = shap.TreeExplainer(model_LGBM)
shap_values = explainer.shap_values(X_test_clean().drop(labels="sk_id_curr", axis=1))

st_shap(shap.summary_plot(shap_values, 
                  feature_names=X_test_clean().drop(labels="sk_id_curr", axis=1).columns,
                  plot_size=(16,12),
                  plot_type="bar",
                  max_display=56,
                  show = False))
plt.title("Interprétation Globale : Diagramme d'Importance des Variables", fontsize=20, fontstyle='italic')
plt.tight_layout()
plt.show()


    # Titre 2
st.markdown("""
                <h1 style="color:#3aa1a2;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                4. Explication du calcul du score de votre client</h1>
                """, 
                unsafe_allow_html=True)
st.write("")

    # Calcul des valeurs Shap
explainer_shap = shap.TreeExplainer(model_LGBM)
shap_values = explainer_shap.shap_values(X_test_clean().drop(labels="sk_id_curr", axis=1))
# récupération de l'index correspondant à l'identifiant du client
idx = int(X_test_clean()[X_test_clean()['sk_id_curr']==ID_client].index[0])

    # Graphique force_plot
st.write("Le graphique suivant appelé `force-plot` permet de connaître où se place la prédiction (f(x)) par rapport à la `base value`.") 
st.write("Nous observons également quelles sont les features qui augmentent la probabilité du client d'être \
            en défaut de paiement (en rouge) et celles qui la diminuent (en bleu), ainsi que l’amplitude de cet impact.")
st_shap(shap.force_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            link='logit',
                            figsize=(20, 8),
                            ordering_keys=True,
                            text_rotation=0,
                            contribution_threshold=0.05))  

 # Graphique decision_plot
st.write("Le graphique ci-dessous appelé `decision_plot` est une autre manière de comprendre la prédiction.\
            Comme pour le graphique précédent, il met en évidence l’amplitude et la nature de l’impact de chaque variable \
            avec sa quantification ainsi que leur ordre d’importance. Mais surtout il permet d'observer \
            “la trajectoire” prise par la prédiction du client pour chacune des valeurs des features affichées. ")

st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            feature_names=X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list(),
                            feature_order='importance',
                            feature_display_range=slice(None, -16, -1), # affichage des 15 variables les + importantes
                            link='logit'))
st.markdown("""
                <br>
                <h1 style="color:#3aa1a2;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                5. Comparaison du profil du client à celui des clients dont la probabilité de défaut de paiement est proche</h1>
                """, 
                unsafe_allow_html=True)
    
# Calcul des valeurs Shap
explainer_shap = shap.TreeExplainer(model_LGBM)
shap_values = explainer_shap.shap_values(X_test_clean().drop(labels="sk_id_curr", axis=1))
shap_values_df = pd.DataFrame(data=shap_values[1], columns=X_test_clean().drop(labels="sk_id_curr", axis=1).columns)
    
df_groupes = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'], shap_values_df], axis=1)
df_groupes['typologie_clients'] = pd.qcut(df_groupes.proba_classe_1,
                                              q=5,
                                              precision=1,
                                              labels=['20%_et_moins',
                                                      '21%_30%',
                                                      '31%_40%',
                                                      '41%_60%',
                                                      '61%_et_plus'])
st.markdown("""
                <h2 style="color:#418b85;text-align:left;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                Comparaison de “la trajectoire” prise par la prédiction du client à celles des groupes de Clients</h2>
                """, 
                unsafe_allow_html=True)
st.write("")

    # Moyenne des variables par classe
df_groupes_mean = df_groupes.groupby(['typologie_clients']).mean()
df_groupes_mean = df_groupes_mean.rename_axis('typologie_clients').reset_index()
df_groupes_mean["index"]=[1,2,3,4,5]
df_groupes_mean.set_index('index', inplace = True)
    
    # récupération de l'index correspondant à l'identifiant du client
idx = int(X_test_clean()[X_test_clean()['sk_id_curr']==ID_client].index[0])

    # dataframe avec shap values du client et des 5 groupes de clients
comparaison_client_groupe = pd.concat([df_groupes[df_groupes.index == idx], 
                                            df_groupes_mean],
                                            axis = 0)
comparaison_client_groupe['typologie_clients'] = np.where(comparaison_client_groupe.index == idx, 
                                                          X_test_clean().iloc[idx, 0],
                                                          comparaison_client_groupe['typologie_clients'])
    # transformation en array
nmp = comparaison_client_groupe.drop(
                      labels=['typologie_clients', "proba_classe_1"], axis=1).to_numpy()

fig = plt.figure(figsize=(8, 20))
st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                                nmp, 
                                feature_names=comparaison_client_groupe.drop(
                                              labels=['typologie_clients', "proba_classe_1"], axis=1).columns.to_list(),
                                feature_order='importance',
                                highlight=0,
                                legend_labels=['Client', '20%_et_moins', '21%_30%', '31%_40%', '41%_60%', '61%_et_plus'],
                                plot_color='inferno_r',
                                legend_location='center right',
                                feature_display_range=slice(None, -21, -1),
                                link='logit'))
st.markdown("""
                <h1 style="color:#3aa1a2;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                6. Graphique de dépendance</h1>
                """, 
                unsafe_allow_html=True)
st.write("Nous pouvons obtenir un aperçu plus approfondi de l'effet de chaque fonctionnalité \
              sur l'ensemble de données avec un graphique de dépendance.")
st.write("Le dependence plot permet d’analyser les variables deux par deux en suggérant une possiblité d’observation des interactions.\
              Le scatter plot représente une dépendence entre une variable (en x) et les shapley values (en y) \
              colorée par la variable la plus corrélées.")

liste_variables = X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list()

col1, col2, = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
with col1:
        ID_var = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                                (liste_variables))
        st.write("Vous avez sélectionné la variable :", ID_var)

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
shap.dependence_plot(ID_var, 
                    valeurs_shap()[1], 
                    X_test_clean().drop(labels="sk_id_curr", axis=1), 
                    interaction_index=None,
                    alpha = 0.5,
                    x_jitter = 0.5,
                    title= "Graphique de Dépendance",
                    ax=ax1,
                    show = False)
ax2 = fig.add_subplot(122)
shap.dependence_plot(ID_var, 
                    valeurs_shap()[1], 
                    X_test_clean().drop(labels="sk_id_curr", axis=1), 
                    interaction_index='auto',
                    alpha = 0.5,
                    x_jitter = 0.5,
                    title= "Graphique de Dépendance et Intéraction",
                    ax=ax2,
                    show = False)
fig.tight_layout()
st.pyplot(fig)
st.markdown("""
                <h1 style="color:#3aa1a2;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                7. Scatter Plots</h1>
                """, 
                unsafe_allow_html=True)
df_2 = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])  
           
fig = px.scatter(df_2, x="proba_classe_1", y="proba_classe_0")
fig.show()
st.markdown("""
                <h1 style="color:#3aa1a2;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                8. Distplots</h1>
                """, 
                unsafe_allow_html=True)
X,Y=st.multiselect("*Veuillez sélectionner deux variable à l'aide du menu déroulant 👇*", 
    (X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list()))
st.write('You selected:', X)
st.write('You selected:', Y)
df = X_test_clean()
fig = px.scatter(df, x=X, y=Y)
fig.show()
