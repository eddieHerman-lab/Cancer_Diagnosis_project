# Cancer_Diagnosis_project
Analise de diagnostico de Câncer de mama

# Diagnóstico de Câncer de Mama com SVM 🩺

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6F61?style=for-the-badge)
![LIME](https://img.shields.io/badge/LIME-00CC66?style=for-the-badge)

Um sistema de apoio a diagnóstico de câncer de mama utilizando Support Vector Machines (SVM) com técnicas avançadas de Machine Learning, explicabilidade (SHAP e LIME) e visualização interativa.

---

## 📌 Visão Geral

Este projeto tem como objetivo classificar tumores mamários como malignos ou benignos com base em características morfológicas nucleares. O sistema utiliza:
- **SVM com kernel RBF** para classificação robusta.
- **SHAP e LIME** para explicabilidade das previsões.
- **Streamlit** para uma interface interativa e amigável.

---

## 🚀 Funcionalidades

- **Classificação Automática**: Previsão de tumores malignos/benignos com alta acurácia.
- **Explicabilidade**: Visualizações SHAP e LIME para entender as decisões do modelo.
- **Análise de Dados**: Visualizações interativas (PCA, UMAP, gráficos de decisão).
- **Otimização de Hiperparâmetros**: Grid Search para ajuste fino do modelo.
- **Interface Amigável**: Aplicação web fácil de usar com Streamlit.

---

## 📊 Métricas de Desempenho

| Métrica               | Valor   |
|-----------------------|---------|
| Acurácia              | 98.2%   |
| AUC-ROC               | 0.995   |
| Sensibilidade         | 97.5%   |
| Especificidade        | 98.8%   |

---

## 🛠️ Como Executar o Projeto

### Pré-requisitos
- Python 3.9+
- Bibliotecas listadas em `requirements.txt`

### Instalação
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/cancer-diagnosis-svm.git
   cd cancer-diagnosis-svm

