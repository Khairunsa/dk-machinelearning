import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('Machine Learning App')

st.info('This is app builds a machine learning model')

with st.expander('Data'):
  st.write('**Raw data**')
  df= pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df
  st.write('**X**')
  X_raw = df.drop('species', axis = 1)
  X_raw
  
  st.write('**Y**')
  Y_raw= df.species
  Y_raw
with st.expander('Data Visualization'):
  #"bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color ='species')

# Input Feature
with st.sidebar:
  st.header('Input Features')
  #"island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  island = st.selectbox('Island',('Biscoe', 'Dream','Torgersen'))
  gender = st.selectbox('Gender',('male', 'female'))
  bill_length_mm = st.slider('Bill length (mm)',32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)

  # Create a Dataframe for the input features
  data = {'island': island,
         'bill_length_mm': bill_length_mm,
         'bill_depth_mm': bill_depth_mm,
         'flipper_length_mm ': flipper_length_mm,
        'body_mass_g': body_mass_g,
         'sex':  gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)
  
with st.expander('Input Featuters'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins
# Data preparation
# Encode x
encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
X = df_penguins [1:]
input_row = df_penguins[:1]
  
# Encode y 
target_mapper = {'Adelie' : 0,
                 'Chinstrap' : 1,
                 'Gentoo':2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (Input Penguin)**')
  input_row
  st.write('**Encoded Y**')
  y

# Model training and inferece
#Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

# Apply model to make prediction
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie','Chinstrap', 'Gentoo']
df_prediction_proba.rename (columns ={ 0: 'Adelie',
                                      1 : 'Chinstrap',
                                      2: 'Gentoo'})
df_prediction_proba
