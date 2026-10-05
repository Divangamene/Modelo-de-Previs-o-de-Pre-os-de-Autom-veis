import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ----------------------------------------------------
st.set_page_config(
    page_title="Winners Group | Sistema de Avaliação Automóvel",
    layout="wide"
)

# ----------------------------------------------------
# ESTILO CORPORATIVO PERSONALIZADO (CSS)
# ----------------------------------------------------
st.markdown("""
    <style>
    /* Fundo Geral e Tipografia Executiva */
    .stApp {
        background-color: #0b0c10;
        color: #e0e6ed;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Cabeçalho Institucional */
    .header-corporativo {
        background: linear-gradient(135deg, #7209b7 0%, #3a0ca3 100%);
        padding: 2.5rem 2rem;
        border-radius: 8px;
        text-align: left;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        border-left: 6px solid #4cc9f0;
    }
    .header-corporativo h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .header-corporativo p {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Bloco de Métricas */
    div[data-testid="stMetric"] {
        background-color: #151821;
        border: 1px solid #282c37;
        padding: 1.2rem;
        border-radius: 6px;
        text-align: left;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* Abas de Navegação Corporativas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #151821;
        padding: 6px;
        border-radius: 6px;
        border: 1px solid #282c37;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 4px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7209b7 !important;
        color: #ffffff !important;
    }

    /* Botão de Ação Principal */
    .stButton>button {
        background-color: #7209b7;
        color: #ffffff;
        border: none;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 4px;
        transition: background-color 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #5c0794;
        color: #ffffff;
    }

    /* Estilização das Tabelas de Dados */
    div[data-testid="stDataFrame"] {
        border: 1px solid #282c37;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# CABEÇALHO DA INSTITUIÇÃO
# ----------------------------------------------------
st.markdown("""
    <div class="header-corporativo">
        <h1>WINNERS GROUP</h1>
        <p>Plataforma de Análise Avançada e Previsão do Valor de Mercado de Um veiculo</p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# CARREGAMENTO DOS DADOS
# ----------------------------------------------------
@st.cache_data
def carregar_dados():
    df = pd.read_csv("dados/carro.csv")
    return df

try:
    dataset = carregar_dados()
    
    # Barra Lateral
    with st.sidebar:
        st.markdown("### **WINNERS GROUP**")
        st.caption("Gestão e Análise Automóvel")
        st.markdown("---")
        st.markdown("**Contactos de Suporte**")
        st.markdown("---")

    # Navegação por Abas
    aba1, aba2, aba3 = st.tabs(["Visão Geral dos Dados", "Análise Estatística", "Cálculo de Avaliação"])

    # ----------------------------------------------------
    # ABA 1: VISÃO GERAL
    # ----------------------------------------------------
    with aba1:
        st.markdown("#### Resumo do Inventário de Dados")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Registos", f"{dataset.shape[0]:,}")
        col2.metric("Atributos Analisados", dataset.shape[1])
        col3.metric("Preço Médio Registado", f"{dataset['Price'].mean():,.2f} €")

        st.markdown("---")
        st.markdown("#### Amostra da Base de Dados")
        st.dataframe(dataset.head(10), use_container_width=True)

        st.markdown("#### Auditoria de Dados Incompletos")
        df_nulos = pd.DataFrame(dataset.isnull().sum(), columns=['Campos Omitidos'])
        st.dataframe(df_nulos.T, use_container_width=True)

    # ----------------------------------------------------
    # TRATAMENTO DOS DADOS
    # ----------------------------------------------------
    df_codificado = dataset.copy()
    encoders = {}
    colunas_categoricas = df_codificado.select_dtypes(include=['object', 'string']).columns

    for coluna in colunas_categoricas:
        le = LabelEncoder()
        df_codificado[coluna] = le.fit_transform(df_codificado[coluna].astype(str))
        encoders[coluna] = le

    # ----------------------------------------------------
    # ABA 2: ANÁLISE ESTATÍSTICA
    # ----------------------------------------------------
    with aba2:
        plt.style.use('dark_background')
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### Matriz de Correlação das Variáveis")
            fig1, ax1 = plt.subplots(figsize=(8, 5.5))
            fig1.patch.set_facecolor('#0b0c10')
            ax1.set_facecolor('#0b0c10')
            
            sns.heatmap(
                df_codificado.corr(), 
                annot=False, 
                cmap="mako", 
                ax=ax1
            )
            st.pyplot(fig1)

        with col_g2:
            st.markdown("#### Distribuição do Valor dos Veículos (até 100.000 €)")
            fig2, ax2 = plt.subplots(figsize=(8, 5.5))
            fig2.patch.set_facecolor('#0b0c10')
            ax2.set_facecolor('#0b0c10')
            
            sns.histplot(
                dataset[dataset['Price'] < 100000]['Price'], 
                kde=True, 
                ax=ax2, 
                color="#7209b7"
            )
            ax2.set_title("Distribuição do Valor de Mercado", color='#e0e6ed', fontsize=10)
            ax2.set_xlabel("Valor em Euros (€)", color='#e0e6ed', fontsize=9)
            ax2.set_ylabel("Frequência de Registos", color='#e0e6ed', fontsize=9)
            st.pyplot(fig2)

    # ----------------------------------------------------
    # ABA 3: CÁLCULO DE AVALIAÇÃO
    # ----------------------------------------------------
    with aba3:
        st.markdown("#### Indicadores de Precisão do Algoritmo")

        X = df_codificado.drop(columns=['Price', 'ID'], errors='ignore')
        y = df_codificado['Price']

        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = DecisionTreeRegressor(random_state=42)
        modelo.fit(X_treino, y_treino)

        y_previsao = modelo.predict(X_teste)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Coeficiente de Determinação (R²)", f"{r2_score(y_teste, y_previsao):.2%}")
        col_m2.metric("Erro Médio Absoluto (MAE)", f"{mean_absolute_error(y_teste, y_previsao):,.2f} €")

        st.markdown("---")
        st.markdown("#### Formulário de Avaliação Veicular")

        dados_entrada = {}
        colunas_form = st.columns(3)
        
        for idx, col in enumerate(X.columns):
            col_destino = colunas_form[idx % 3]
            if col in colunas_categoricas:
                opcoes = list(dataset[col].unique())
                valor_selecionado = col_destino.selectbox(f"{col}", opcoes)
                valor_codificado = encoders[col].transform([str(valor_selecionado)])[0]
                dados_entrada[col] = valor_codificado
            else:
                valor_padrao = float(dataset[col].median())
                valor_num = col_destino.number_input(f"{col}", value=valor_padrao)
                dados_entrada[col] = valor_num

        st.write("")
        if st.button("Calcular Estimativa de Valor"):
            df_entrada = pd.DataFrame([dados_entrada])
            resultado = modelo.predict(df_entrada)[0]
            
            st.markdown(f"""
                <div style="background-color: #151821; 
                            border: 1px solid #7209b7; 
                            padding: 1.5rem; 
                            border-radius: 6px; 
                            text-align: center; 
                            margin-top: 1.5rem;">
                    <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Valor Estimado de Mercado</span>
                    <h2 style="margin: 0.5rem 0 0 0; color: #ffffff; font-size: 2.2rem; font-weight: 700;">{resultado:,.2f} €</h2>
                </div>
            """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Ficheiro `dados/carro.csv` não encontrado. Certifique-se de que a diretoria e o ficheiro estão devidamente salvos.")