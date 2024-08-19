import streamlit as st
import pandas as pd
import math
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Configuração inicial da aplicação
st.set_page_config(page_title="NBA Data Analysis", layout="wide")

# Função para carregar os dados
@st.cache_data
def load_data(url):
    data = pd.read_csv(url)
    return data

# Título da aplicação
st.title("NBA Data Analysis - Exploratory Data and Regression")

# Introdução
st.markdown("""
## **Introdução**
Neste projeto, exploramos um conjunto de dados da NBA para entender como a performance dos jogadores evolui ao longo do tempo e como diferentes fatores influenciam suas carreiras. A aplicação permite explorar os dados, visualizar tendências e aplicar modelos de regressão linear para fazer previsões.
""")

# Seção para carregar os dados
st.header("Carregamento de Dados")
url = "https://raw.githubusercontent.com/caio-santt/datascience-course---NBA-database/main/all_seasons.csv"
st.markdown("### Base de dados")
st.write("Os dados serão carregados a partir da URL fornecida.")

# Carregando os dados
data = load_data(url)
st.write("Dados carregados com sucesso! Aqui estão as 5 primeiras linhas do dataset:")
st.dataframe(data.head())

# Análise Exploratória de Dados (EDA)
st.header("Análise Exploratória de Dados (EDA)")

# Exibir colunas do dataset
st.write("Colunas disponíveis no dataset:", data.columns.tolist())

# Selecionar colunas para análise
selected_columns = st.multiselect("Selecione as colunas para análise:", data.columns.tolist())
if selected_columns:
    st.write("Resumo estatístico das colunas selecionadas:")
    st.write(data[selected_columns].describe())

    st.write("Boxplots das colunas selecionadas:")
    fig, ax = plt.subplots()
    sns.boxplot(data=data[selected_columns], ax=ax)
    st.pyplot(fig)

    st.write("Histogramas das colunas selecionadas:")
    fig, ax = plt.subplots()
    data[selected_columns].hist(ax=ax, bins=30)
    st.pyplot(fig)

# Modelagem de Regressão Linear
st.header("Modelagem de Regressão Linear")

# Seleção de Variáveis
x_var = st.selectbox("Selecione a variável independente (X):", data.columns.tolist())
y_var = st.selectbox("Selecione a variável dependente (Y):", data.columns.tolist())

if x_var and y_var:
    X = data[[x_var]]
    Y = data[y_var]

    # Divisão dos dados
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Modelo de Regressão
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Resultados
    st.write(f"Coeficiente de Regressão: {model.coef_[0]}")
    st.write(f"Intercepto: {model.intercept_}")
    st.write(f"R²: {r2_score(Y_test, Y_pred)}")
    st.write(f"MSE: {mean_squared_error(Y_test, Y_pred)}")

    # Visualização da Regressão
    fig, ax = plt.subplots()
    ax.scatter(X_test, Y_test, color='blue', label='Dados Reais')
    ax.plot(X_test, Y_pred, color='red', label='Regressão Linear')
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.legend()
    st.pyplot(fig)

# Tendências de Peso, Altura, Idade e Diversidade
st.header("Tendências ao Longo do Tempo")

# Seleção de Variável de Interesse
trend_var = st.selectbox("Selecione a variável para análise de tendência:", ['height', 'weight', 'age', 'nationality'])

if trend_var:
    if trend_var != 'nationality':
        # Análise de Tendência
        st.write(f"Analisando a tendência de {trend_var} ao longo do tempo.")
        trend_data = data.groupby('season_start')[trend_var].mean()

        # Gráfico de Tendência
        fig, ax = plt.subplots()
        ax.plot(trend_data.index, trend_data.values, marker='o', linestyle='-', color='green')
        ax.set_title(f"Tendência de {trend_var} ao longo do tempo")
        ax.set_xlabel("Temporada")
        ax.set_ylabel(trend_var.capitalize())
        st.pyplot(fig)
    else:
        # Diversidade Geográfica
        st.write("Analisando a diversidade geográfica ao longo do tempo.")
        data['nationality_grouped'] = data['nationality'].apply(lambda x: 'Americano' if x == 'USA' else 'Não-Americano')
        diversity_data = data.groupby('season_start')['nationality_grouped'].value_counts(normalize=True).unstack()

        # Gráfico de Diversidade
        fig, ax = plt.subplots()
        diversity_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title("Diversidade Geográfica ao Longo do Tempo")
        ax.set_xlabel("Temporada")
        ax.set_ylabel("Proporção")
        st.pyplot(fig)
