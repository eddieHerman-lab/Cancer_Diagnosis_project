# Cancer_Diagnosis_project
Analise de diagnostico de Câncer de mama

Nesta análise, utilizamos o Support Vector Machine (SVM) para a classificação de tumores malignos e benignos com base no conjunto de dados do Breast Cancer Wisconsin. Consideramos diferentes abordagens, incluindo o SVM linear e o SVM com kernel RBF, para entender a separabilidade dos dados e a melhor forma de modelá-los.

O dataset de câncer de mama do scikit-learn contém características extraídas de imagens digitalizadas de aspirados por agulha fina (FNA) de massas mamárias. Cada instância representa medidas de características de células no tecido mamário. A variável alvo indica se o tecido é maligno (0) ou benigno (1). As características incluem medidas como raio, textura, perímetro, área, suavidade, compacidade, concavidade, etc. Essas medidas são calculadas para o núcleo celular e podem indicar anomalias associadas ao câncer. Importância: Entender seu dataset é fundamental. Neste caso, trabalhando com um problema médico real onde o objetivo é classificar corretamente tumores como malignos ou benignos. Falsos negativos (classificar erroneamente um tumor maligno como benigno) podem ter consequências graves.
O projeto tambem possui um arquivo que funciona como um weapp de interface iterativa feito e Streamlit para demontrções

