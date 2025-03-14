# Cancer_Diagnosis_project
Analise de diagnostico de Câncer de mama

# Diagnóstico de Câncer de Mama com SVM 🩺

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6F61?style=for-the-badge)
![LIME](https://img.shields.io/badge/LIME-00CC66?style=for-the-badge)

Um sistema de apoio a diagnóstico de câncer de mama utilizando Support Vector Machines (SVM) com técnicas avançadas de Machine Learning, explicabilidade (SHAP e LIME) e visualização interativa.
Este projeto tem como objetivo utilizar técnicas de Machine Learning para classificar tumores mamários em benignos e malignos com base no dataset "Breast Cancer Wisconsin". A abordagem principal utiliza Support Vector Machines (SVM) combinadas com calibração de probabilidades para fornecer previsões mais confiáveis, essenciais em contextos clínicos.

## 📌 Visão Geral

O projeto explora os seguintes aspectos:

Pré-processamento dos dados: Normalização, tratamento de valores faltantes e balanceamento de classes.
Redução de Dimensionalidade: Uso de PCA para visualização inicial da separabilidade das classes.
Modelagem com SVM: Treinamento de modelos com kernel linear e kernel RBF.
Calibração de Probabilidades: Aplicação de CalibratedClassifierCV para ajustar as previsões probabilísticas, melhorando a confiabilidade dos resultados.
Validação e Interpretação: Avaliação dos modelos por meio de métricas (acurácia, precisão, recall, F1-score) e visualizações (matriz de confusão, histogramas e boxplots).

Metodologia

1. Coleta e Preparação dos Dados
   
Dataset: Breast Cancer Wisconsin (disponível no scikit-learn).
Pré-processamento: Normalização das features e uso de class_weight='balanced' para mitigar problemas de desbalanceamento.

4. Visualização com PCA
Redução para 2 componentes para explorar a estrutura dos dados e identificar possíveis sobreposições entre as classes.

6. Treinamento dos Modelos SVM
SVM Linear: Modelo inicial para avaliar a separabilidade linear.
SVM com Kernel RBF: Modelo não-linear que melhor captura a complexidade dos dados, evidenciado por métricas superiores.
8. Calibração das Probabilidades
Aplicação de CalibratedClassifierCV para transformar as saídas do decision_function do SVM em probabilidades que reflitam melhor a chance real de pertencer à classe positiva, essencial para decisões clínicas.

10. Validação e Análise dos Resultados
    
Uso de validação cruzada estratificada para garantir a robustez dos resultados.
Avaliação por meio de métricas de classificação (acurácia, precisão, recall, F1-score) e visualizações (matriz de confusão, histogramas de probabilidades e boxplots).
12. Implementação de uma Interface Interativa (Opcional)
Criação de um web app interativo com Streamlit para permitir a visualização dos dados e a realização de predições em tempo real.
Como Reproduzir o Projeto
Clone o repositório:

bash
Copiar
Editar
git clone https://github.com/eddieHerman-lab/Cancer_Diagnosis_project.git
cd Cancer_Diagnosis_project
Instale as dependências:

bash
Copiar
Editar
pip install -r requirements.txt
Execute o notebook:

Abra o notebook SVM_project_Tumores_mamarios_predicao.ipynb no Google Colab ou em sua IDE local para reproduzir as análises.
(Opcional) Execute o web app com Streamlit:

Certifique-se de que o modelo treinado foi salvo (ex.: svm_model.pkl).
Rode o aplicativo:
bash
Copiar
Editar
streamlit run app.py
Resultados e Discussão
Métricas de Desempenho: Os modelos SVM com kernel RBF obtiveram uma acurácia de 97%, com precisão, recall e F1-score superiores, indicando que a abordagem não-linear se adapta melhor à complexidade dos dados.
Calibração de Probabilidades: A calibração ajustou as previsões para refletirem probabilidades mais extremas (próximas a 0 ou 1), demonstrando maior convicção na classificação, o que é crucial em diagnósticos médicos.
Visualizações: Gráficos de PCA, histogramas e boxplots auxiliaram na identificação da estrutura dos dados e no diagnóstico da separabilidade das classes.
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

