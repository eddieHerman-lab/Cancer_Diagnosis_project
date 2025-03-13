import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Carregando o dataset de câncer de mama
data = load_breast_cancer()
X = data.data
y = data.target

# Convertendo para DataFrame para melhor visualização
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Visualizando as primeiras linhas do dataset
print(df.head())

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualizando a distribuição das classes
sns.countplot(x='target', data=df)
plt.title("Distribuição das Classes")
plt.show()

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Aplicando PCA para reduzir para 2 componentes e visualizar os dados
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='coolwarm', edgecolor='k')
plt.title("Visualização dos Dados com PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()



# 1. SVM com kernel linear
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)

y_pred_linear = svm_linear.predict(X_test)

print("=== SVM Linear ===")
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# 2. SVM com kernel RBF
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)

y_pred_rbf = svm_rbf.predict(X_test)

print("=== SVM RBF ===")
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# ANÁLISE EXPLORATÓRIA MAIS DETALHADA
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Matriz de Correlação")
plt.show()

# Visualização da distribuição das características por classe
features = df.columns[:-1]
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features[:10]):  # Primeiras 10 características
    plt.subplot(2, 5, i + 1)
    sns.boxplot(x='target', y=feature, data=df)
plt.tight_layout()
plt.show()

# VALIDAÇÃO CRUZADA E OTIMIZAÇÃO DE HIPERPARÂMETROS
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Definindo os parâmetros para o GridSearchCV
param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    'kernel': ['rbf']
}

param_grid_linear = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear']
}

# Criando o StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV para SVM RBF
grid_rbf = GridSearchCV(SVC(random_state=42), param_grid_rbf, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
grid_rbf.fit(X_train, y_train)

print("Melhores parâmetros para SVM RBF:", grid_rbf.best_params_)
print("Melhor score para SVM RBF:", grid_rbf.best_score_)

# GridSearchCV para SVM Linear
grid_linear = GridSearchCV(SVC(random_state=42), param_grid_linear, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
grid_linear.fit(X_train, y_train)

print("Melhores parâmetros para SVM Linear:", grid_linear.best_params_)
print("Melhor score para SVM Linear:", grid_linear.best_score_)

# Avaliando o melhor modelo
best_model = grid_rbf.best_estimator_
y_pred = best_model.predict(X_test)

print("=== Melhor Modelo SVM ===")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# VISUALIZAÇÃO DA FRONTEIRA DE DECISÃO
from matplotlib.colors import ListedColormap


def plot_decision_boundary(X, y, model, title):
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.show()


# Aplicar PCA nos dados de treino
X_train_pca = pca.transform(X_train)

# Treinar modelos com dados PCA
svm_rbf_pca = SVC(kernel='rbf', C=grid_rbf.best_params_['C'],
                  gamma=grid_rbf.best_params_['gamma'], random_state=42)
svm_rbf_pca.fit(X_train_pca, y_train)

svm_linear_pca = SVC(kernel='linear', C=grid_linear.best_params_['C'], random_state=42)
svm_linear_pca.fit(X_train_pca, y_train)

# Visualizar fronteiras de decisão
plot_decision_boundary(X_train_pca, y_train, svm_rbf_pca, "Fronteira de Decisão - SVM RBF")
plot_decision_boundary(X_train_pca, y_train, svm_linear_pca, "Fronteira de Decisão - SVM Linear")

# CURVA ROC e AUC
# Predizendo probabilidades para calcular ROC
y_pred_proba = best_model.decision_function(X_test)

# Calculando a curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotando a curva ROC
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# COMPARAÇÃO COM OUTROS ALGORITMOS
# Definindo os modelos para comparação
models = {
    'SVM (RBF)': best_model,
    'SVM (Linear)': grid_linear.best_estimator_,
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Resultados para cada modelo
results = {}

for name, model in models.items():
    if name not in ['SVM (RBF)', 'SVM (Linear)']:  # Estes já foram treinados
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    results[name] = accuracy

# Plotando a comparação
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.xlabel('Algoritmo')
plt.ylabel('Acurácia')
plt.title('Comparação de Algoritmos')
plt.xticks(rotation=45)
plt.ylim(0.90, 1.0)  # Ajuste conforme necessário
plt.show()
