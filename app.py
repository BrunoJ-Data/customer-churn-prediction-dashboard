import streamlit as st
import pandas as pd
import plotly.express as px


import joblib

st.set_page_config(layout="wide")

st.subheader(' 🧠 ML Churn Prediction Dashboard : [Saas Subscription]')

st.write("Analyse predictive du risque départ clients : Plateforme d'abonnement Informatique ( secteur, revenu ,pays)")
st.divider()



#--------------------------------------------------------------

#f_final.csv ->  ->filtres --> KPI ->preprocessing ML ->prédiction


@st.cache_data  # cache pour ne pas avoir à recharger  CSV , requetes SQL:
def load_data():
    df = pd.read_csv('model/df_final.csv')
    return df

 # cache  pour garder en mémoire le model:
@st.cache_resource
def load_model():
    model = joblib.load('model/model_lr.joblib') 
    return model  

 # cache  pour garder en mémoire le scaler: # @st.cache_resource recommandé par Streamlit pour les objets ML
@st.cache_resource
def load_scaler():
    scaler = joblib.load('model/scaler.joblib') 
    return scaler 


# chargement  des fichiers:

df_final = load_data()
model_lr = load_model()
scaler = load_scaler()

#-------------------------------------------------------
## BARRE LATERALE
#--------------------------------------------------------
st.sidebar.header("🔍 Filtres de l'Audit")

# classement list par ordre alphabétique
liste_pays = sorted(df_final['country'].unique())

select_countries = st.sidebar.multiselect( '🌍 Country:',options= liste_pays ) #liste des pays cs la colonne 'country'

#Onglet type de contrat
type_contrat = sorted(df_final['contrat_type'].unique()) ## .unique() ajouté pour n'avoir qu'une seule fois chaque type de contrat

select_contract = st.sidebar.multiselect('📄 Contrat:', options= type_contrat)


type_industry =  sorted(df_final['industry'].unique())

select_industry =  st.sidebar.multiselect('💻Industry:',options = type_industry )

#--------------------------------
#### ___ FILTRAGE                                                     (ancienneté)
#--------------------------------
# INITIALISER  le slider /calcules une fois les bornes min/max  absolu du jeu de données:
min_seniority_days = int(df_final['seniority_days'].min())
max_seniority_days = int(df_final['seniority_days'].max())



slider_senior_clients = st.sidebar.slider('📅 Filter Clients (in days)',
                                  min_value = df_final['seniority_days'].min(),
                                  max_value = df_final['seniority_days'].max(),  
                                  value=(min_seniority_days, max_seniority_days) )

st.sidebar.caption(' <- New clients _____ old clients ->')

#  Le slider renvoie un tuple (min, max),  On le décompose dans deux variables plus explicites :  # min_days = borne minimale et # max_days = borne maximale chooisi par l'user
min_days, max_days = slider_senior_clients

#-------------------------------------------------
# Filtres DataFrame — Tous les .isin regroupés
#---------------------------------------------------

# 1__"Je garde uniquement les clients qui sont entre min_days et max_days du slider !!!.
df_filtre = df_final[
    (df_final["seniority_days"] >= min_days)&
    (df_final["seniority_days"] <= max_days) ]

                
# Filtre Pays

# .isin() = teste si la valeur de la colonne appartient à la liste sélectionnée (True/False)

if select_countries :   # Si je selectionne un pays de 'select countries'...
    df_filtre= df_filtre[df_filtre['country'].isin(select_countries)] #  
    #Je garde seulement les lignes dont le pays fait partie des pays sélectionnés



#_ de je reduit une 3eme fois  le nombre de clients avec le nombre de contrats selectionner


# Filtre contrat 
if select_contract: # Si je selectionne un contrat 
    df_filtre = df_filtre[df_filtre['contrat_type'].isin(select_contract)]  # Je mets à jour df_filtre : je garde seulement les lignes dont le contrat fait partie des contrats sélectionnés

# Filtre Industry
if select_industry :
    df_filtre = df_filtre[df_filtre['industry'].isin(select_industry)] 
    # Je garde seulement les lignes dont l'industrie fait partie des industries sélectionnées


#st.write(df_filtre.shape)

if df_filtre.empty:
    st.write('* No clients - Give more details')
    st.stop()

#--------------------------------------

#Pre Process ML
#( Recreer le PREPROCESS qui ont permis de mettre en place les features 

#Le reindex sert à garantir ça.

#_____________

df_ml = df_filtre.copy() ### separe pour le ml

#--------------------------------------------------------------------

#Numeriser les colonnes textuelles
df_ml = pd.get_dummies(df_ml, columns= ['billing_frequency','industry','country','referral_source','contrat_type' ], drop_first= True )###


# recréer la séparation
X= df_ml.drop(columns =['churn_flag','account_id'])

#  le modele s'attend à avoir tout les colonnes apprisent lors du fit, sinon il plante
# pour  palier à ça, lors de la selection des colonnes,  on utilise le .reindex ,il permet de créer les colonnes manquantes rempli avec les valeurs 0
# reindex  doit etre appliquer uniquement sur  X , qui contient uniquement les features du modèle
# X dataFrame ML , doit être fait dans l’ordre du scaler.features_names_in_
# 

X = X.reindex(columns= scaler.feature_names_in_, fill_value = 0) #Recréer les colonnes manquantes


#Toujours reindexer AVANT de réordonner.
#Reindex = recréer les colonnes manquantes (tjrs le X)




#j'applique la mise à l'echelle (avec le scale) sur le 'X_scaled':
X_scaled = scaler.transform(X)

#predict_proba ne renvoie pas une prédiction, mais des probabilités.
y_proba = model_lr.predict_proba(X_scaled) 


# On récupère les probalitéspour chaques clients  || predict_proba() retourne 2 colonnes ) 0: non chrun ,  1:  churn stocker ds une variable
churn_proba = y_proba[:,1]

## je rajoute une colonne proba au DataFrame
df_filtre['churn_proba'] = churn_proba

#--------------------------------------------
# KPI
#--------------------------------------------

mrr_filtre = df_filtre["current_mrr"].sum() #somme de la colonne MRR
#--------
#nbre_clients_filtrer = len(df_filtre )

#total_risque : MRR a risque
mrr_chaque_client =  df_filtre["current_mrr"]

# risque churn clients
risque_client = mrr_chaque_client*churn_proba
mrr_a_risque = risque_client.sum()


# Taux de churn client:
pourcent_churn = df_filtre['churn_proba'].mean()*100




KPI1_col1, KP2_col2, KPI3_col3 = st.columns(3)


#st.write() → pour les DataFrames, shapes, objets Python

#st.markdown() → pour styliser tes KPI (taille + vert fluo + formatage)




KPI1_col1.write("**MRR** [revenu mensuel]") 
KPI1_col1.markdown(f"# `{mrr_filtre:,.0f}$`") #utilisation des backticks :“affiche ce texte comme du code inline"


KP2_col2.write(" **Loss risk** [risque de perte]")
KP2_col2.markdown(f"# `{mrr_a_risque:,.0f}$`")

KPI3_col3.write("**Churn** [Estimation départ clients]")
KPI3_col3.markdown(f"# `{pourcent_churn:.2f}%`")

#--------------------------------------------------------------------
#GRAPHIQUE
#-----------------------------------------------------------------------

st.divider()
#Graphique_a :
df_filtre['mrr_a_risque'] = df_filtre['current_mrr'] * df_filtre['churn_proba']


# reset_index remet les index 0,1,etc... apres le 'groupby' 'Industry' est mis  en index, donc on remet tout a la normale...
grafik_a = df_filtre.groupby('industry').agg({'mrr_a_risque':'sum','churn_proba':'mean'}).reset_index()

grafik_a['tx_churn'] = ( grafik_a['churn_proba']*100).round(2)

#st.dataframe(grafik_a)

fig_col1,space_col,fig_col2 = st.columns([4, 1, 4])

fig_a = px.bar(
    grafik_a,
    x='industry',
    y='tx_churn',
    text='tx_churn',
    title= 'Customer Churn Risk by Industry %',
    color ='industry'
)

fig_a.update_layout(showlegend=False) #Modifications (masquer la légende + formater les chiffres)
fig_a.update_traces(texttemplate='%{text:.1f}%', textposition='outside')


# affichage dans Streamlit
fig_col1.plotly_chart(fig_a, use_container_width=True)



# Graphique_b :

grafik_b = df_filtre.groupby('contrat_type').agg({'mrr_a_risque':'sum'}).reset_index() #si je ne mets pas reset_index contrat type devient l'index

fig_b = px.pie(
    grafik_b,
    names='contrat_type',
    values='mrr_a_risque',
    title= "Revenue at Risk by Contract Type",
    hole=0.6,
    color='contrat_type',
    color_discrete_map=({
        'Enterprise': "#AB0CF5",   
        'Pro':"#F5F10E",          
        'Basic':"#1CA301"        
    }
)
    
)

fig_b.update_layout(
    annotations=[dict(  #annotation attend un dictionnaire
        text="Contract",
        x=0.5,
        y=-0.28,
        showarrow=False
    )]
)

fig_col2.plotly_chart(fig_b, use_container_width=True)

#----
grafik_c = df_filtre.groupby('industry').agg({'mrr_a_risque':'sum','churn_proba':'mean'}).reset_index()

fig_c = px.bar(
    grafik_a,
    x='industry',
    y='tx_churn',
    text='tx_churn',
    title= 'Risque départ clients par Industrie %',
    color ='industry'
)


st.divider()
#---------------------------------------------------
#st.write(df_filtre['churn_proba'].describe()) #pour repérer lee Quartiles

def risk_level(lignes):
    if lignes >= 0.58:
        return('🔴 High  ')
    elif lignes >= 0.18:
        return('🟠 Medium  ')
    else:
        return('⚪ Low ')


df_filtre['departure_risk'] = df_filtre['churn_proba'].apply(risk_level)

#-------------------
 
# Graphique:
st.write('Dashboard clients :')


#st.write(df_filtre[['tickets_nb', 'total_errors', 'satisfaction_avg']].describe()) # pour avoir les quartiles 

#ajout colonne: explicative churn

def review_warning(client):
    warning_high= []

    if client['total_errors'] > 34:
        warning_high.append('Too many errors')

    if  client['tickets_nb'] >5:
            warning_high.append('Too many tickets')

    if client['satisfaction_avg'] <3.67:
        warning_high.append('Low satisfaction')

    if warning_high: 
        return " • ".join(warning_high)
    else:
        return ''


df_filtre['Risk_Factors'] = df_filtre.apply(review_warning, axis= 1) #axis 1 applique la fonction ligne par ligne qd on a plusuieurs colonnes



# Colore la probabilité de churn du vert (faible risque) au rouge (fort risque)
st.dataframe(
    df_filtre[['account_id','current_mrr','satisfaction_avg','total_errors','country','churn_proba','departure_risk','Risk_Factors']]
        .style #e veux transformer ce DataFrame en Styler, pour pouvoir le mettre en forme visuellement.
        .format({'churn_proba': '{:.2f}', 'satisfaction_avg': '{:.2f}','tickets_nb':'{:.2f}','current_mrr':'{:,.0f} $'})# sert à transformer des données brutes en un texte propre ,ajouter des symboles ($, %), de forcer un nombre de décimales, ou d'ajouter des séparateurs de milliers
        .background_gradient(subset=['churn_proba'], cmap='Reds'),
    use_container_width=True,
    column_config={
        "Risk_Factors": st.column_config.TextColumn("Risk Factors", width="medium")
    }
)



        

#-------------------------------
st.divider()


fig_col3,space_col,fig_col4 = st.columns([4, 1, 4])


model_lr = joblib.load("model/model_lr.joblib")

# Importance = coefficients absolus
importances = abs(model_lr.coef_[0])

# DataFrame
df_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=True)

# Graphique Plotly
fig_c = px.bar(
    data_frame=df_importance,
    x='importance',
    y='feature',
    orientation='h',
    title='Churn Drivers [Facteurs qui déclenchent le départ]'
)
fig_c.update_traces(marker_color='#f77348')  # Mets le code HEX de ton choix
fig_col3.plotly_chart(fig_c, use_container_width=True) #Étire le graphique pour qu’il prenne toute la largeur disponible.


#Graphique c:


#client_risque = pd.DataFrame({'client':'account_id',
#'probabilité':'churn_proba'}, index=[0]) # définis l’index de la ligne que tu crées

#st.write(pd.DataFrame(client_risque))

#________________________


#dataframe_scat= st.dataframe(df_filtre[['current_mrr','satisfaction_avg']])
satisfation_data = pd.DataFrame({'Customer MRR': df_filtre['current_mrr'],
                                'satisfaction':df_filtre['satisfaction_avg'],
                                'Churn risk':df_filtre['churn_proba'] })

fig_d = px.scatter(
    satisfation_data,
    x='Customer MRR',
    y='satisfaction',
    title='Customer Satisfaction ',
    color='Churn risk',
    
    color_continuous_scale=[
        '#ffffff',
        "#fd0000",
    ],  # Dégradé : Blanc (0) vers Rouge (1)
)

fig_col4.plotly_chart(fig_d, use_container_width=True)

