# Core Pkgs
import streamlit as st
import numpy as np
import pandas as pd
import os, glob
# Separation du jeu de donnée
from sklearn.model_selection import train_test_split
# Metrique
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
# Erreur
import warnings
warnings.filterwarnings("ignore")

st.title("Prédiction sur le statut du client de la banque")

def file_selector(folder_path='C:/Users/Charly/Documents/Tp/'):
	    filenames = os.listdir(folder_path)
	    selected_filename = st.selectbox('Select a file', filenames)
	    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)
df = pd.read_csv(filename, sep=';')
st.write(df.head())

# required features for predicting churning customers
required_fields = ["Attrition_Flag", "Contacts_Count_12_mon", "Total_Trans_Ct", "Total_Trans_Amt",
                   "Total_Revolving_Bal", "Avg_Utilization_Ratio"]
df = df.loc[:, required_fields]

def get_dataset(df):
    y = df['Attrition_Flag']
    X = df.drop('Attrition_Flag',errors='ignore',axis=1)
    return X, y
X, y = get_dataset(df)

# required features for user input
required_fields = ["Contacts_Count_12_mon", "Total_Trans_Ct", "Total_Trans_Amt",
                   "Total_Revolving_Bal", "Avg_Utilization_Ratio"]
st.write('Caractéristiques requises (abrégé et dans l\'ordre):', required_fields)

st.subheader("Veuillez renseignez les valeurs ci-dessous en chiffres")

with st.form(key='form1'):
    f1 = st.number_input("Nombre de contacts au cours des 12 derniers mois",1,100)
    f2 = st.number_input("total de transactions (12 derniers mois)",1,100)
    f3 = st.number_input("Montant total de la transaction (12 derniers mois)",1,100000)
    f4 = st.number_input("Solde renouvelable total sur la carte de crédit",1,3000)
    f5 = st.number_input("Taux d'utilisation moyen de la carte",0,1)
    user_input = np.array([f1,f2,f3,f4,f5]).reshape(1, -1)
    submit_button = st.form_submit_button(label='Evaluer')

customers_type = df.groupby("Attrition_Flag")
# 1. Clearly distinguish between chunked and un-chunked customers (grouping).

churned_customers = customers_type.get_group("Attrited Customer")
unchurned_customers = customers_type.get_group("Existing Customer")

# 2. Use factors from our data exploration that have clearly seperated 
# chunked from un-chunked customers to select distinctive un-chunked customers.

dist_churned_customers = unchurned_customers.where(
    (df.Total_Trans_Ct > 85) &
    (df.Total_Trans_Amt > 5000) &
    ((df.Total_Revolving_Bal > 500) & (df.Total_Revolving_Bal < 2500)) &
    (df.Avg_Utilization_Ratio > 0.1)
).dropna()

non_dist_indexes = [i for i in list(df.index) if i not in list(dist_churned_customers.index)]
non_dist_churned_customers = unchurned_customers.reindex(index=non_dist_indexes).dropna()

#Clean dataset, and convert appropriate columns to appropriate dtypes()  

# convert bool and object to category 
cat_types = ['bool','object','category']
data_clean = df.copy()
data_clean[data_clean.select_dtypes(cat_types).columns] = data_clean.select_dtypes(cat_types).apply(lambda x: x.astype('category'))
st.text("Dataset utilsé pour la prédiction : ")
st.write(data_clean.head()) 
    # Label and One Hot Encoding on catagorical independent variables
    # Label Encoding for ordinal variables (ex: rankings, scales, etc)
    # One Hot Encoding for nominal variables (ex: color, gender, etc.)

    # Use One Hot Eoncoding on each catagorical independent variables 
    # Because Income_Category has an "Unknown" value, unable to convert to an ordinal variable to use Label Encoding 

    # https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python

# Split data_clean into two datasets y - depedent variable, x - independent variables 
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



# Calculate Precision, Recall, F1-Score, and Accurarcy 
# https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9 
# predictions using RFC model given X_test data
y_pred = RFC.predict(X_test)

user_input_pred = RFC.predict(user_input)
statut_client = ""
if user_input_pred == 0:
    statut_client = "Existing"
else:
    statut_client = "Attrited"

if submit_button:
    st.success("Le statut du client est {} ".format(statut_client))

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
# precision, recall, f1-score
precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1,beta = 1)

# classification_report for Attrited Customer  
st.write("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred)*100.0))
st.write("Recall: %.2f%%" % ((recall_score(y_test,y_pred))*100.0))



# else:
#     erreur = st.text_input("Erreur de traitement !")
	






# if __name__ == '__main__':
# 	main()