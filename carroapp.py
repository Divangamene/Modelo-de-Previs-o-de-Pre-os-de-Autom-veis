import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Configurações da página
st.set_page_config(
    page_title="Sistema de Previsão de Preço de Veículos",
    layout="wide"
)

# Estilo personalizado simples
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Painel de Análise e Previsão de Preços de Veículos</div>', unsafe_allow_html=True)

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("dados/carro.csv")
    return df

try:
    dataset = carregar_dados()
    
    # Abas de navegação
    aba1, aba2, aba3 = st.tabs(["Visão Geral dos Dados", "Análise Gráfica", "Modelo de Previsão"])

    # ----------------------------------------------------
    # ABA 1: VISÃO GERAL
    # ----------------------------------------------------
    with aba1:
        st.subheader("Informações do Conjunto de Dados")
        
        # Métricas principais
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Registos", dataset.shape[0])
        col2.metric("Total de Atributos", dataset.shape[1])
        col3.metric("Preço Médio", f"{dataset['Price'].mean():,.2f} €")

        st.markdown("---")
        st.write("### Primeiras Linhas da Tabela de Dados")
        st.dataframe(dataset.head(10), use_container_width=True)

        st.write("### Verificação de Valores Em Falta (Nulos)")
        df_nulos = pd.DataFrame(dataset.isnull().sum(), columns=['Valores em Falta'])
        st.dataframe(df_nulos.T, use_container_width=True)

    # ----------------------------------------------------
    # PREPARAÇÃO E TRATAMENTO DOS DADOS
    # ----------------------------------------------------
    df_codificado = dataset.copy()
    encoders = {}
    
    colunas_categoricas = df_codificado.select_dtypes(include=['object', 'string']).columns

    for coluna in colunas_categoricas:
        le = LabelEncoder()
        df_codificado[coluna] = le.fit_transform(df_codificado[coluna].astype(str))
        encoders[coluna] = le

    # ----------------------------------------------------
    # ABA 2: ANÁLISE GRÁFICA
    # ----------------------------------------------------
    with aba2:
        st.subheader("Matriz de Correlação das Variáveis")
        
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        sns.heatmap(df_codificado.corr(), annot=False, cmap="coolwarm", ax=ax1)
        st.pyplot(fig1)

        st.subheader("Distribuição dos Preços dos Veículos")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(dataset[dataset['Price'] < 100000]['Price'], kde=True, ax=ax2, color="#2C3E50")
        ax2.set_title("Distribuição de Preços (Veículos até 100.000 €)")
        ax2.set_xlabel("Preço (€)")
        ax2.set_ylabel("Frequência")
        st.pyplot(fig2)

    # ----------------------------------------------------
    # ABA 3: MODELO E PREVISÃO
    # ----------------------------------------------------
    with aba3:
        st.subheader("Métricas do Modelo (Árvore de Decisão)")

        X = df_codificado.drop(columns=['Price', 'ID'])
        y = df_codificado['Price']

        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = DecisionTreeRegressor(random_state=42)
        modelo.fit(X_treino, y_treino)

        y_previsao = modelo.predict(X_teste)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Coeficiente de Determinação (R²)", f"{r2_score(y_teste, y_previsao):.4f}")
        col_m2.metric("Erro Médio Absoluto (MAE)", f"{mean_absolute_error(y_teste, y_previsao):,.2f} €")

        st.markdown("---")
        st.subheader("Simulação de Preço por Atributos")

        # Formulário de entrada de dados
        dados_entrada = {}
        colunas_form = st.columns(3)
        
        for idx, col in enumerate(X.columns):
            col_destino = colunas_form[idx % 3]
            if col in colunas_categoricas:
                opcoes = list(dataset[col].unique())
                valor_selecionado = col_destino.selectbox(f"Seleccionar {col}", opcoes)
                valor_codificado = encoders[col].transform([str(valor_selecionado)])[0]
                dados_entrada[col] = valor_codificado
            else:
                valor_padrao = float(dataset[col].median())
                valor_num = col_destino.number_input(f"Introduzir {col}", value=valor_padrao)
                dados_entrada[col] = valor_num

        if st.button("Calcular Estimativa de Preço", type="primary"):
            df_entrada = pd.DataFrame([dados_entrada])
            resultado = modelo.predict(df_entrada)[0]
            st.info(f"### Valor Estimado do Veículo: **{resultado:,.2f} €**")

except FileNotFoundError:
    st.error("Ficheiro `dados/carro.csv` não encontrado. Certifique-se de que a pasta `dados` e o ficheiro `carro.csv` estão no mesmo diretório do programa.")