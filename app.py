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
# Erreur
import warnings
warnings.filterwarnings("ignore")



# Initialize connection.
# Uses st.cache to only run once.
# @st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
# def init_connection():
#     return mysql.connector.connect(**st.secrets["mysql"])

# conn = init_connection()

# Perform query.
# Uses st.cache to only rerun when the query changes or after 10 min.
# @st.cache(ttl=600)
# def run_query(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         return cur.fetchall()

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

def select_aleatoire(nbre):
    c.execute('SELECT Contacts_Count_12_mon, Total_Trans_Ct, Total_Trans_Amt, Total_Revolving_Bal, Avg_Utilization_Ratio , Attrition_Flag FROM data_credit_card ORDER BY RANDOM() LIMIT "{}"'.format(nbre))
    data = c.fetchall()
    return data

menu = ["Unique prediction","Multiple prediction from the database","About"]
choice = st.sidebar.selectbox("Préedictions options",menu)

def file_selector(folder_path='./database/'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)
df = pd.read_csv(filename, sep=';')

# required features for predicting churning customers
required_fields = ["Attrition_Flag", "Contacts_Count_12_mon", "Total_Trans_Ct", "Total_Trans_Amt",
                "Total_Revolving_Bal", "Avg_Utilization_Ratio"]
df_model = df.loc[:, required_fields]

#Clean dataset, and convert appropriate columns to appropriate dtypes()  
# convert bool and object to category 
cat_types = ['bool','object','category']
data_clean = df_model.copy()
data_clean[data_clean.select_dtypes(cat_types).columns] = data_clean.select_dtypes(cat_types).apply(lambda x: x.astype('category'))

# Label and One Hot Encoding on catagorical independent variables
# https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python

# Split data_clean into two datasets y - depedent variable, x - independent variables 
# Map Attrited Customer = 1 and Existing Customer = 0
codes = {'Existing Customer':1, 'Attrited Customer':0}
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

scaler = RobustScaler()

#  train-test stratified split using 80-20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0, shuffle= True,stratify = y)
# Initial fit using RandomForestClassifier
RFC = RandomForestClassifier(random_state = 0)
RFC.fit(X_train,y_train)
 
# predictions using RFC model given X_test data
y_pred = RFC.predict(X_test)

if choice == "Unique prediction":
    st.subheader("Select Banque.csv")
    st.title('Unique prediction of the bank customer\'s status')
    st.subheader("Please fill in the values ​​below in numbers")

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
    if submit_button:
        st.success("The customer's status is {} ".format(statut_client))

elif choice == "Multiple prediction from the database":
    number = st.number_input('Enter a positive number between 10127', min_value=0, max_value=10127, step=1)
    # rows = run_query(f"SELECT Contacts_Count_12_mon, Total_Trans_Ct, Total_Trans_Amt, Total_Revolving_Bal, Avg_Utilization_Ratio FROM `banque` ORDER BY RAND() LIMIT {number};")
    # st.title("Prédiction multiple depuis la base de données")
    # mult_stat_client = []
    # for row in rows:
    #     db_input = np.array(row).reshape(1, -1)
    #     st.write(db_input)
    #     mult_rand_db_input_pred = RFC.predict(db_input)
    #     # st.write(mult_rand_db_input_pred)
    #     if  mult_rand_db_input_pred == 0:
    #         mult_stat_client.append('Existing')
    #     else:
    #         mult_stat_client.append('Attrited')
    #     # st.write(mult_stat_client)
    # pred_db_input = DataFrame(mult_stat_client,columns= ['Prediction'])
    # st.write(pred_db_input)

    rechercher = st.button('Load')
    if rechercher:
        resul_rech_al = select_aleatoire(number)
        result_al_df = pd.DataFrame(resul_rech_al,columns=["Contacts_Count_12_mon", "Total_Trans_Ct", "Total_Trans_Amt",
                "Total_Revolving_Bal", "Avg_Utilization_Ratio","Attrition_Flag"] )
        # st.dataframe(result_al_df)

        liste_pred = []
        for i in range(result_al_df.shape[0]):
            X = result_al_df.iloc[i,:-1]
        #    for j in range(X.shape[0]):
        #     #    st.write(X[j])
            X=np.array(X).reshape(1,-1)
            #st.write(X)

            ypred = RFC.predict(X)
            # st.write(ypred)

            if ypred==0:
                msg="Existing client"
            elif ypred==1:
                msg="Attrited client"

            liste_pred.append(msg)
        # st.dataframe(pd.DataFrame(liste_pred, columns=["Predictions"]))

        de=pd.concat([result_al_df, pd.DataFrame(liste_pred, columns=["Predictions"])], axis=1)
        st.dataframe(de)
else:
    st.title("About")
    st.subheader("CONTEXT")
    st.text("A bank official wants to reduce the number of customers who leave their credit card services. He would like to be able to anticipate the departure of customers in order to provide them with better services and thus retain them.")
    st.subheader("GOALS")
    st.text("Set up a Machine Learning model capable of predicting customer departures trained from the Banque.csv dataset.")
    st.text('Prédiction multiple depuis la base de données: donne le statut d\'un nombre sélectionnés de clients de la banque selon les valeurs enregistrés en base de données')
    st.text('Dataset original :')
    st.write(df.head())
    st.write('Required features (abrégé et dans l\'ordre):', required_fields)
    st.subheader("Dataset description :")
    st.write("Initial dimension of the dataset : ", df.shape)
    st.write("Number of class to predict : ", len(np.unique(df['Attrition_Flag'])))
    st.write("Differents classes : ", np.unique(df['Attrition_Flag']))
    st.text("Example of a dataset sample use to predict : ")
    st.write(data_clean.head()) 
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    # precision, recall, f1-score
    precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1,beta = 1)

    # classification_report for Attrited Customer  
    st.write("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred)*100.0))
    st.write("Recall: %.2f%%" % ((recall_score(y_test,y_pred))*100.0))

    st.success("Built with Streamlit")
    st.info("LinkedIn : Charles Emmanuel Kouadio")

    


# if __name__ == '__main__':
# 	main()