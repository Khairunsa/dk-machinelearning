import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Halaman Utama
st.set_page_config(page_title="Machine Learning - Penguin Species", layout="wide")
st.title('Machine Learning App for Penguin Species Prediction')

st.markdown("""<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .info-box {
        background-color: #f4f6f8;
        border-radius: 8px;
        padding: 10px;
    }
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: bold;
        color: #34495e;
    }
</style>
<div class="main-title">Machine Learning App</div>
""", unsafe_allow_html=True)

st.info('This app predicts penguin species using a machine learning model created by Khairunnisa Aulia Rahma.')
st.caption("Contact: khairunnisaauliarahma7785@gmail.com")

with st.expander('Data'):
  st.write('**Raw data**')
  df= pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df
  st.write('**X**')
  X_raw = df.drop('species', axis = 1)
  X_raw
  
  st.write('**Y**')
  Y_raw = df.species
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

y = Y_raw.apply(target_encode)

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

#Display prediction species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba, column_config={'Adelie': st.column_config.ProgressColumn(
  'Adelie', format='%f', width='medium', min_value=0, max_value=1),
  'Chinstrap': st.column_config.ProgressColumn(
  'Chinstrap', format='%f', width='medium', min_value=0, max_value=1),
  'Gentoo': st.column_config.ProgressColumn(
  'Gentoo', format='%f', width='medium', min_value=0, max_value=1),}, hide_index=True)
  
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
