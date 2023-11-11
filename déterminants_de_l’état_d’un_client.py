# -*- coding: utf-8 -*-

# Importing libraries


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Uploading the dataset

df = pd.read_csv('Cas_étude_1_Clients_banque.csv')
df.head()

""" Essayeons de voir le pourcentage des valeurs manquantes, valeurs uniques, ainsi que le type de chaque feature(variable)"""

stats = []
for col in df.columns:
    stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].dtype))
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', '% MissingValues', 'type'])
    df_ = stats_df.sort_values('% MissingValues', ascending=False)
df_

"""On peut remarquer que heureusement il n'y a pas aucune valeur manquante dans notre dataset et que chaque colonne est dans son type approprié.

On doit quand meme supprimer certains colonne irrelevant comme "RowNumber","CustomerId",
"""

df = df.drop(columns = ['RowNumber', "CustomerId", "Surname"], axis=1)

"""Maintenant Visualizons notre variable cible "Exited" pour voir le pourcentage de chaque categorié"""

plt.hist(df['Exited'])
plt.title('Distribution des clients')
plt.xlabel('Catégorie')
plt.ylabel('Fréquence')
plt.show()

df['Exited'].values

churned = df[df['Exited']==1]['Exited'].count()
not_churned = df[df['Exited']==0]['Exited'].count()
labels  = ['Churned', 'Not churned']
array = [churned, not_churned]
plt.pie(array, labels = labels,autopct='%1.1f%%')
plt.title('Distribution des clients')
plt.show()

"""# Séparer les variables spécifiques de la variable cible"""

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X

"""# Encoding categorical data"""

X.head()

from sklearn.preprocessing import LabelEncoder

categorical_columns = [ 'Geography', 'Gender']

encoder = LabelEncoder()
for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column])

X

"""## La mise à échelle des variables"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
columns_to_scale = ['CreditScore', 'Balance', 'EstimatedSalary']

X[columns_to_scale] = sc.fit_transform(X[columns_to_scale])

"""# Train Test Split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()

print(f"the shape of X_train : {X_train.shape}")
print(f"the shape of X_test : {X_test.shape}")
print(f"the shape of y_train : {y_train.shape}")
print(f"the shape of y_test : {y_test.shape}")

"""#  modèle de la régression logistique"""

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(class_weight='balanced')

log_reg.fit(X_train, y_train)

coefficients = log_reg.coef_
intercept = log_reg.intercept_

print(f" Coefficients {coefficients}")
print(f"Intercept {intercept}")

"""Pour interpréter les estimations des paramètres du modèle et quelles sont les variables les plus déterminantes de l’action du quitter la banque on va suivre ces regles:
* Plus un coefficient est positif, plus la variable correspondante a un impact positif sur la probabilité de quitter la banque.
* Plus un coefficient est négatif, plus la variable correspondante a un impact négatif sur la probabilité de quitter la banque.
* La valeur absolue d'un coefficient indique l'importance relative de la variable dans le modèle. donc les variables avec une grande valuer absolue sont plus déterminants pour le modele.

Dans ce cas Les variables "Age" , "Gender", "Balance" ont un impact plus que les autres variables

## Random Forest
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy {accuracy}")

"""# Metrics"""

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

y_pred_logistic = log_reg.predict(X_test)

cm_logistic = confusion_matrix(y_test, y_pred_logistic)
print("Matrice de confusion (Régression Logistique):")
print(cm_logistic)

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Accuracy (Régression Logistique):", accuracy_logistic)

auc_logistic = roc_auc_score(y_test, y_pred_logistic)
print("AUC (Régression Logistique):", auc_logistic)

print('--------------------------------------------')
# Pour le Random Forest

y_pred_rf = rf_classifier.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Matrice de confusion (Random Forest):")
print(cm_rf)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy (Random Forest):", accuracy_rf)

auc_rf = roc_auc_score(y_test, y_pred_rf)
print("AUC (Random Forest):", auc_rf)

"""C/C :  le Random Forest présente une meilleure accuracy globale,et aussi une meilleure capacité de discrimination (mesurée par l'AUC)

### Prévoir, à l’aide des deux modèles, si le client ci-dessus va quitter la banque
"""

client_data = pd.DataFrame({
    'CreditScore': [555],
    'Geography': [2],
    'Gender': [0],
    'Age': [25],
    'Tenure': [6],
    'Balance': [120000],
    'NumOfProducts': [1],
    'HasCrCard': [0],
    'IsActiveMember': [1],
    'EstimatedSalary': [30000]
})

client_data

"""Scale it first"""

client_data[columns_to_scale] = sc.transform(client_data[columns_to_scale])

prediction_logistic = log_reg.predict(client_data)

prediction_rf = rf_classifier.predict(client_data)

print(f"Prédiction avec la Régression Logistique: {prediction_logistic}")
print(f"Prédiction avec le Random Forest: {prediction_rf}")

"""D'après les prédictions des deux modèles, le client ne va pas churner, c'est-à-dire qu'il va rester avec la banque."""

