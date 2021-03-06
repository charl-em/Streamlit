# Core Pkgs
import streamlit as st
import numpy as np
import pandas as pd
from pandas import DataFrame
import os, glob
# Connexion à la base de données
import sqlite3
# Normalisation
from sklearn.preprocessing import RobustScaler
# Separation du jeu de donnée
from sklearn.model_selection import train_test_split
# Metrique
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
# File Processing Pkgs
import pandas as pd
# import docx2txt
# from PyPDF2 import PdfFileReader
# import pdfplumber

from PIL import Image

# Erreur
import warnings
warnings.filterwarnings("ignore")

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

def select_aleatoire(nbre):
    c.execute('SELECT Contacts_Count_12_mon, Total_Trans_Ct, Total_Trans_Amt, Total_Revolving_Bal, Avg_Utilization_Ratio , Attrition_Flag FROM data_credit_card ORDER BY RANDOM() LIMIT "{}"'.format(nbre))
    data = c.fetchall()
    return data

menu = ["About","Unique prediction","Multiple prediction from the database","Select from a file"]
choice = st.sidebar.selectbox("Prediction options",menu)

def file_selector(folder_path='./database/'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        
        return os.path.join(folder_path, selected_filename)

filename = file_selector()
df = pd.read_csv(filename, sep=';')
st.write("You selected :open_file_folder: `%s`" % filename)
# required features for predicting churning customers
required_fields = ["Attrition_Flag", "Contacts_Count_12_mon", "Total_Trans_Ct", "Total_Trans_Amt",
                "Total_Revolving_Bal", "Avg_Utilization_Ratio"]
df_model = df.loc[:, required_fields]

#Clean dataset, and convert appropriate columns to appropriate dtypes()  
# convert bool and object to category 
cat_types = ['bool','object','category']
data_clean = df_model.copy()
data_clean[data_clean.select_dtypes(cat_types).columns] = data_clean.select_dtypes(cat_types).apply(lambda x: x.astype('category'))

# Map Attrited Customer = 1 and Existing Customer = 0
codes = {'Existing Customer':0, 'Attrited Customer':1}
data_clean['Attrition_Flag'] = data_clean['Attrition_Flag'].map(codes)

y = data_clean['Attrition_Flag']
X = data_clean.drop('Attrition_Flag',errors='ignore',axis=1) 

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

features_to_encode = X.select_dtypes('category').columns.to_list()
for feature in features_to_encode:
    X = encode_and_bind(X, feature)

#  train-test stratified split using 80-20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0, shuffle= True,stratify = y)
# Initial fit using RandomForestClassifier
RFC = RandomForestClassifier(random_state = 0)
RFC.fit(X_train,y_train)
 
# predictions using RFC model given X_test data
y_pred = RFC.predict(X_test)

## FONCTION POUR MANIPULER LE DATAFRAME
def edit_dataframe(data):
    ## Pour selectionner des lignes et souvegarder les lignes sélectionnées
    update_mode_value = GridUpdateMode.MODEL_CHANGED

    gb = GridOptionsBuilder.from_dataframe(data)

    # customize gridOptions
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)

    gb.configure_selection(selection_mode='multiple', use_checkbox=True, groupSelectsChildren=True)

    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
   
    grid_response = AgGrid(data,
                           gridOptions=gridOptions,
                           width='100%',
                           update_mode=update_mode_value,
                           enable_enterprise_modules='Enable Enterprise Modules'
                           )

    ## Les valeurs sélectionnées de mon Dataframe
    df = grid_response['data']

    selected = grid_response['selected_rows']
    selected_df = pd.DataFrame(selected) 
    return selected,selected_df

## Lire et predire tous le dataset
def read_predict(df):
    colonne_selectionne = ["CLIENTNUM",
                           "Contacts_Count_12_mon",
                           "Total_Trans_Ct",
                           "Total_Trans_Amt",
                           "Total_Revolving_Bal",
                           "Avg_Utilization_Ratio","Attrition_Flag"]
    liste_bool = []
    for col in colonne_selectionne:
        if col in df.columns:
            liste_bool.append(True)
        else:
            liste_bool.append(False)

    if all(liste_bool) == False:
        st.warning("Operation not possible, the file you loaded does not contain adequate columns for the prediction.")
    else:
        data = df[colonne_selectionne]
        selected, selected_df = edit_dataframe(data) 
        X = selected_df.drop(["CLIENTNUM","Attrition_Flag"], axis=1)
        ypred = RFC.predict(X)
        liste_pred = []
        for pred in ypred:
            if pred == 0:
                msg = "Existing client"
            else:
                msg = "Attrited client"

            liste_pred.append(msg)

        st.subheader("Predictions of selected clients:")

        de = pd.concat([selected_df, pd.DataFrame(liste_pred, columns=["Predictions"])], axis=1)
        st.dataframe(de) # AgGrid(de)

if choice == "Unique prediction":
    st.subheader("Select Banque.csv")
    st.title('Unique prediction of the bank customer\'s status')
    st.subheader("Please fill in the values below in numbers")

    with st.form(key='form1'):
        f1 = st.number_input("Number of contacts in the last 12 months",1,100)
        f2 = st.number_input("total transactions (last 12 months)",1,100)
        f3 = st.number_input("Total amount of the transaction (last 12 months)",1,100000)
        f4 = st.number_input("Total revolving balance on the credit card",1,100000)
        f5 = st.number_input("Average card usage rate",0,1)
        user_input = np.array([f1,f2,f3,f4,f5]).reshape(1, -1)
        submit_button = st.form_submit_button(label='Assess')

    user_input_pred = RFC.predict(user_input)
    statut_client = ""
    if user_input_pred == 0:
        statut_client = "Existing"
    else:
        statut_client = "Attrited"
    if submit_button and statut_client == "Existing":
        st.success("The customer's status is {} ".format(statut_client))
        st.balloons()
    else:
        st.warning("The customer's status is {} ".format(statut_client))

elif choice == "Multiple prediction from the database":
    number = st.number_input('Enter a positive number between 10127', min_value=0, max_value=10127, step=1)
    rechercher = st.button('Load')
    if rechercher:
        resul_rech_al = select_aleatoire(number)
        result_al_df = pd.DataFrame(resul_rech_al,columns=["Contacts_Count_12_mon", "Total_Trans_Ct", "Total_Trans_Amt",
                "Total_Revolving_Bal", "Avg_Utilization_Ratio","Attrition_Flag"] )
        liste_pred = []
        for i in range(result_al_df.shape[0]):
            X = result_al_df.iloc[i,:-1]
        #    for j in range(X.shape[0]):
        #     #    st.write(X[j])
            X=np.array(X).reshape(1,-1)
            ypred = RFC.predict(X)
            if ypred==0:
                msg="Existing client"
            elif ypred==1:
                msg="Attrited client"

            liste_pred.append(msg)
        de=pd.concat([result_al_df, pd.DataFrame(liste_pred, columns=["Predictions"])], axis=1)
        st.dataframe(de)

elif choice == "Select from a file":
    st.title("Select from a file some clients based on their features")
    with st.expander('View Data'):
        AgGrid(df)
    st.subheader("Select one or multiple rows")
    if df.empty == False:
        df = pd.read_csv(filename, sep=';')
        read_predict(df)
    else:
        df = pd.read_table(filename, sep=';')
        read_predict(df)
else:
    st.title("About")
    image = Image.open('bank.jpg')
    st.image(image)
    st.header("CONTEXT :smirk:")
    st.markdown("A bank official wants to reduce the number of customers who leave their credit card services. He would like to be able to anticipate the departure of customers in order to provide them with better services and thus retain them.")
    st.header("GOALS :sunglasses:")
    st.markdown("Set up a Machine Learning model capable of predicting customer departures trained from the Banque.csv dataset.")
    st.markdown('**Unique prediction**: predict the statut of a client based on the features filled in the form')
    st.markdown('**Multiple prediction from the database**: gives the status of a selected number of bank customers according to the values ​​recorded in the database')
    st.text('Dataset original : ')
    st.write(df.head())
    st.write('Required features (abrégé et dans l\'ordre): :eyes:', required_fields)
    st.subheader("Dataset description :")
    st.write("Initial dimension of the dataset : ", df.shape)
    st.write("Number of class to predict : ", len(np.unique(df['Attrition_Flag'])))
    st.write("Differents classes : ", np.unique(df['Attrition_Flag']))
    st.write('0 : :x: 1 : :heavy_check_mark:')
    st.text("Example of a dataset sample use to predict : ")
    st.write(data_clean.head()) 
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    # precision, recall, f1-score
    precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1,beta = 1)

    # classification_report for Attrited Customer  
    st.write("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred)*100.0))
    st.write("Recall: %.2f%%" % ((recall_score(y_test,y_pred))*100.0))

    st.info("LinkedIn : Charles Emmanuel Kouadio :wave:")