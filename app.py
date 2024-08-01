import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar o modelo treinado
model = joblib.load('data/titanic_survival_model.pkl')

# Título da aplicação
st.title('Previsão de Sobrevivência no Titanic')

# Entrada de dados do usuário
Pclass = st.slider('Classe', 1, 3, 2)
Sex = st.radio('Sexo', options=['Homem', 'Mulher'])
Age = st.number_input('Idade', min_value=0, max_value=100, value=30)
SibSp = st.slider('Número de irmãos/cônjuges a bordo', 0, 8, 0)
Parch = st.slider('Número de pais/filhos a bordo', 0, 6, 0)
Fare = st.number_input('Tarifa', min_value=0.0, max_value=1000.0, step=10.0)

# Mapear para o valor interno usado pelo modelo
input_sex = 'male' if Sex == 'Homem' else 'female'

# Botão para previsão
if st.button('Prever Sobrevivência'):
    # Preparar os dados de entrada
    input_data = pd.DataFrame([[Pclass, input_sex, Age, SibSp, Parch, Fare]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    
    # Fazer a previsão
    prediction = model.predict(input_data)
    
    # Exibir o resultado
    if prediction[0] == 1:
        st.write('O passageiro provavelmente sobreviveria.')
    else:
        st.write('O passageiro provavelmente não sobreviveria.')
