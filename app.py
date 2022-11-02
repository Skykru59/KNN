import numpy as  np
import sklearn as svm
import seaborn as sns
import missingno
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib.pyplot import axis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

df = pd.read_csv("iris.csv")

X = df [['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df ['species']
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
knn.score(x_test ,y_test)


st.title('choisir son iris')
sp_length = st.sidebar.slider('sepal_length', 4.0, 8.0)
sp_width = st.sidebar.slider('sepal_width', 2.0, 5.0)
pt_length = st.sidebar.slider('petal_length', 1.0, 8.0)
pt_width = st.sidebar.slider('petal_width', 0.0, 3.0)

data = {
    "sepal_length" : [sp_length],
    "sepal_width" : [sp_width],
    "petal_length" : [pt_length],
    "petal_width" : [pt_width],
    "species" : "unknow"
}

data_utilisateur = [sp_length, sp_width, pt_length, pt_width]
espece_utilisateur = knn.predict([data_utilisateur])

df_choix_utilisateur = pd.DataFrame(data)
df_final = pd.concat([df, df_choix_utilisateur], axis = 0)

if st.button("Appuyez ici ou pas !!!"):
    st.success("pas de chance vous êtes :" + espece_utilisateur[0])
    st.pyplot(sns.pairplot(df_final, x_vars=["petal_length"], y_vars=["petal_width"], hue="species", markers=["o", "s", "D", "p"]))
    st.pyplot(sns.pairplot(df_final, x_vars=["sepal_length"], y_vars=["sepal_width"], hue="species", markers=["o", "s", "D", "p"]))
    st.write("try again !")