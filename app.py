import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn


df = pickle.load(open('df.pkl','rb'))

st.title("Titanic Survival Classifier")

Sex = st.selectbox('Sex',df['sex'].unique())

age = st.number_input('Age')

fare = st.number_input('Fare')

classs = st.selectbox('Class' , df['class'].unique())

embarktown = st.selectbox('Embark Town' , df['embark_town'].unique())
family = st.number_input('No of People Travelling with passengers')

from sklearn.model_selection import train_test_split
X  = df.iloc[: , 1:7]
Y = df.iloc[: , 0].values
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3,random_state=115)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[
    ('tnf1' , StandardScaler() , ['age','fare']),
    ('tnf2' , OrdinalEncoder(categories=[['Third','Second','First']]) , ['class']),
    ('tnf3' , OneHotEncoder(sparse=False , drop='first'),['sex','embark_town'])
],remainder='passthrough')


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
X_train_transformed = transformer.fit_transform(X_train)
transformer.fit_transform(X_test)
lr = LogisticRegression()
lr.fit(X_train_transformed , Y_train)
data = transformer.transform(pd.DataFrame(columns=['sex', 'age', 'fare', 'class', 'embark_town', 'family']
                                    , data=np.array([Sex, age, fare, classs, embarktown, family]).reshape(1, 6)))
lr.predict(data)

if st.button('Predict'):
    var = lr.predict(data)[0]
    if var == 1:
        st.title("Passenger Survived")

    else:
        st.title("Passenger not survived")






