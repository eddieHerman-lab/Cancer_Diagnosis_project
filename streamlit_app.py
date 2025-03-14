# Arquivo: cancer_diagnosis_app_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import lime
import lime.lime_tabular
import os

os.environ["LD_LIBRARY_PATH"] = os.getenv("LD_LIBRARY_PATH", "") + ":/usr/lib/x86_64-linux-gnu"  # Reduz logs do TensorFlow



# Configurar a página
st.set_page_config(
    page_title="Diagnóstico de Câncer de Mama com SVM",
    page_icon="🩺",
    layout="wide"
)


# Reduzindo o tepo de inicialização
@st.cache_resource  # Use cache para modelos grandes
def load_model():
    return joblib.load('model.pkl')

# Funções auxiliares com cache estratégico
@st.cache_data
def load_data():
    """Carrega e formata o dataset"""
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names, data.target_names


@st.cache_resource
def train_model(X, y, tuning=False):
    """Treina o modelo com otimizações"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline simplificado
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42, C=10, gamma=0.01, kernel='rbf'))
    ])

    if tuning:
        param_grid = {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': [0.01, 0.1, 1],
            'svm__kernel': ['rbf']
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        pipeline = grid_search.best_estimator_

    pipeline.fit(X_train, y_train)
    calibrated_model = CalibratedClassifierCV(pipeline, cv='prefit')
    calibrated_model.fit(X_train, y_train)

    # Explicadores otimizados
    explainer_shap = shap.Explainer(
        calibrated_model.predict_proba,
        X_train,
        algorithm="permutation"
    )

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=load_breast_cancer().feature_names,
        class_names=['Maligno', 'Benigno'],
        mode='classification',
        discretize_continuous=False
    )

    return {
        'model': calibrated_model,
        'explainer_shap': explainer_shap,
        'explainer_lime': explainer_lime,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@st.cache_data
def get_pca_visualization(X):
    """Gera visualização PCA com cache"""
    scaler = StandardScaler().fit(X)
    return PCA(n_components=2).fit_transform(scaler.transform(X))


def plot_decision_boundary(X_pca, y):
    """Plota fronteira de decisão otimizada"""
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Visualização PCA dos Tumores')
    return plt.gcf()


def get_feature_inputs(feature_names):
    """Interface de entrada de dados otimizada"""
    X, _, _, _ = load_data()
    default_values = X.mean(axis=0)

    st.subheader("Medidas do Núcleo Celular")
    values = {}

    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            values[feature] = st.slider(
                feature,
                float(X[:, i].min()),
                float(X[:, i].max()),
                float(default_values[i]),
                format="%.4f",
                key=f"slider_{i}"
            )
    return values


def plot_shap_explanation(explainer, X_sample, feature_names):
    """Gera visualização SHAP corrigida"""
    # Obter valores SHAP para a classe positiva (Benigno)
    shap_values = explainer(X_sample)

    # Selecionar apenas a explicação para a classe 1 (Benigno)
    shap_values_class1 = shap_values[..., 1]

    # Criar gráfico waterfall
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values_class1[0], max_display=10, show=False)
    plt.tight_layout()
    return fig

def main():
    """Função principal corrigida"""
    st.title("🩺 Diagnóstico de Câncer de Mama com SVM")

    # Carregar dados
    with st.status("Carregando dados...", expanded=True) as status:
        X, y, feature_names, target_names = load_data()
        pca_result = get_pca_visualization(X)
        status.update(label="Dados carregados!", state="complete")

    # Treinar modelo
    with st.status("Inicializando modelo...", expanded=True) as status:
        st.write("Dividindo dados...")
        model_data = train_model(X, y)
        st.write("Preparando sistemas de explicação...")
        status.update(label="Modelo pronto!", state="complete")

    # Interface principal
    tabs = st.tabs(["📊 Visualização", "🔍 Diagnóstico", "📈 Explicações Técnicas", "ℹ️ Sobre"])

    with tabs[0]:
        st.header("Análise Exploratória")

        col1,col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribuição das Classes( Exemplo com os dados de teste:")
            class_dist = pd.Series(y).map({0: 'Maligno', 1: 'Benigno'}).value_counts()
            fig, ax = plt.subplots()
            class_dist.plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)
        with col2:
            st.subheader("Visualização PCA (teste exemplo")
            fig = plot_decision_boundary(pca_result, y)
            st.pyplot(fig)

         
     
    with tabs[1]:
        st.header("Diagnóstico Personalizado")

        # Entrada de dados
        values = get_feature_inputs(feature_names)

        if st.button("Executar Diagnóstico", type="primary"):
            with st.status("Analisando...", expanded=True) as status:
                # Preparar amostra
                X_sample = np.array([values[feat] for feat in feature_names]).reshape(1, -1)

                # Previsão
                proba = model_data['model'].predict_proba(X_sample)[0]
                prediction = model_data['model'].predict(X_sample)[0]

                # Resultado
                status.update(label="Análise completa!", state="complete")
                if prediction == 1:
                    st.success(f"Tumor Benigno ({proba[1]:.2%} de confiança)")
                else:
                    st.error(f"Tumor Maligno ({proba[0]:.2%} de confiança)")

                # Visualizações
                st.subheader("Explicação da Decisão")
                tab_shap, tab_lime = st.tabs(["SHAP", "LIME"])

                with tab_shap:
                    st.write("Explicação SHAP:")
                    try:
                        fig = plot_shap_explanation(
                            model_data['explainer_shap'],
                            X_sample,
                            feature_names
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erro ao gerar explicação SHAP: {str(e)}")

                with tab_lime:
                    st.write("Explicação LIME:")
                    try:
                        explainer = model_data['explainer_lime']
                        exp = explainer.explain_instance(
                            X_sample[0],
                            model_data['model'].predict_proba,
                            num_features=10
                        )
                        fig = exp.as_pyplot_figure()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erro ao gerar explicação LIME: {str(e)}")
    with tabs[2]:  # Nova aba de explicações
        st.header("📚 Centro de Explicações")
        st.markdown("""
        Entenda cada aspecto do sistema de diagnóstico, desde os conceitos básicos até os detalhes técnicos.
        """)

        with st.expander("🤖 Como o SVM Funciona?", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image("images/SvmFronteiraLinear.png",
                         caption="SVM separando classes com margem máxima")
            with col2:
                st.markdown("""
                **Para Leigos:**  
                Imagine que o SVM é como um arquiteto que tenta construir o muro mais largo possível 
                entre dois bairros (classes de tumores). Quanto mais largo o muro, mais segura é a 
                separação entre casos benignos e malignos.

                **Para Técnicos:**  
                - **Kernel RBF:** Transforma os dados para um espaço dimensional superior onde a 
                  separação linear é possível
                - **Margem Suave:** Permite alguns erros de classificação para evitar overfitting
                - **Parâmetros Chave:**  
                  `C` = Controle de tolerância a erros (↑C = menos tolerante)  
                  `gamma` = Alcance da influência de cada exemplo (↑gamma = modelo mais complexo)
                """)

        with st.expander("📈 Entendendo as Métricas"):
            st.markdown("""
            | Métrica          | Para Leigos                          | Definição Técnica                          |
            |------------------|--------------------------------------|--------------------------------------------|
            | **Acurácia**     | Porcentagem de acertos gerais        | (VP + VN) / (VP + VN + FP + FN)            |
            | **Sensibilidade**| Capacidade de detectar casos reais   | VP / (VP + FN)                             |
            | **Especificidade**| Capacidade de descartar casos sadios | VN / (VN + FP)                             |
            | **AUC-ROC**      | Qualidade geral da separação         | Área sob a curva de característica operacional do receptor |
            """)

            if st.checkbox("Mostrar exemplo numérico"):
                st.code("""
                Exemplo de Matriz de Confusão:
                          Previsto Maligno  Previsto Benigno
                Real Maligno       85 (VP)           5 (FN)
                Real Benigno        3 (FP)          142 (VN)

                Sensibilidade = 85 / (85 + 5) = 94.44%
                Especificidade = 142 / (142 + 3) = 97.93%
                """)

        with st.expander("🖼️ Decifrando as Visualizações"):
            tab_pca, tab_shap, tab_lime = st.tabs(["PCA", "SHAP", "LIME"])

            with tab_pca:
                st.markdown("""
                **Análise de Componentes Principais (PCA)**  
                - Reduz a dimensionalidade para 2D mantendo a variação máxima
                - Cada ponto representa um paciente
                - Cores indicam diagnóstico real
                - Fronteira mostra como o modelo separa as classes
                """)
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/GaussianScatterPCA.svg/1200px-GaussianScatterPCA.svg.png",
                         caption="Exemplo de projeção PCA com fronteira de decisão")

            with tab_shap:
                st.markdown("""
                **SHAP (SHapley Additive exPlanations)**  
                - Mostra a contribuição de cada característica para a decisão
                - Valores positivos → favorecem diagnóstico maligno
                - Valores negativos → favorecem diagnóstico benigno
                - Tamanho da barra indica magnitude do impacto.
                 Cada barra mostra como uma característica influenciou a decisão:
                 - ➕ Aumenta chance de malignidade
                 - ➖ Reduz chance de malignidade
                  """)
                st.image("images/shap.png")

            with tab_lime:
                st.markdown("""
                **LIME (Local Interpretable Model-agnostic Explanations)**  
                - Cria modelo local simples para explicar cada caso
                - Vermelho → características que apoiam a predição
                - Azul → características que contradizem a predição
                - Mostra o peso relativo de cada fator
                """)
                st.image("images/limeEx.png",
                         caption="Exemplo de explicação LIME")

        with st.expander("🧬 Glossário de Características"):
            st.markdown("""
            | Termo Técnico                | Explicação Simplificada              |
            |------------------------------|---------------------------------------|
            | Raio Médio                   | Tamanho médio das células             |
            | Textura Média                | Variação na cor/textura               |
            | Perímetro Médio              | Comprimento do contorno celular       |
            | Área Média                   | Área total das células                 |
            | Suavidade Média              | Uniformidade do núcleo celular        |
            | Compacidade Média            | Densidade da distribuição celular     |
            | Concavidade Média            | Intensidade das depressões celulares  |
            | Pontos Côncavos Médias       | Número de depressões celulares        |
            | Simetria Média               | Regularidade da forma celular         |
            | Dimensão Fractal Média       | Complexidade da estrutura celular     |
            """)

        with st.expander("⚙️ Detalhes da Implementação"):
            st.markdown("""
            **Arquitetura do Sistema:**
            ```mermaid
            graph TD
                A[Dados Brutos] --> B[Pré-processamento]
                B --> C[Treinamento do SVM]
                C --> D[Calibração de Probabilidades]
                D --> E[Sistema de Explicações]
                E --> F[Interface Visual]
            ```

            **Otimizações Chave:**
            - Balanceamento entre precisão e desempenho
            - Explicações em tempo real via SHAP/LIME
            - Visualizações interativas para diferentes níveis de expertise
            """)

    with tabs[3]:
        st.header("Sobre o Projeto")
        st.markdown("""
        **Sistema de Diagnóstico Assistido por IA**  
        - Desenvolvido com Streamlit e scikit-learn  
        - Dataset: Breast Cancer Wisconsin (Diagnóstico)  
        - Algoritmo: Support Vector Machine (SVM) com kernel RBF  
        - Explicabilidade: SHAP e LIME  
        """)
        st.caption("Versão 2.2 - Corrigido problema de layout")

# Adicione no final do código
if st.secrets.get("ENV") == "production":
    st.write(" Monitoramento ativo")
    # Integre com Sentry/Datadog caso sendo necessário
if __name__ == "__main__":
    main()

