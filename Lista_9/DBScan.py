import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from kneed import DataGenerator, KneeLocator #para mostrar o número de grupos ideal do agrupamento
from sklearn.cluster import KMeans #Importando a função Kmeans
from sklearn.preprocessing import StandardScaler #Função utilizada para normalização dos dados
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler #Função utilizada para normalização dos dados
import sys

import pickle
with open('./creditcard.pkl', 'rb') as f:
  X_treino, X_teste, y_treino, y_teste = pickle.load(f)

from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

# 1. Configurar o modelo
# eps: Distância máxima entre dois pontos para serem considerados vizinhos.
# min_samples: Mínimo de pontos para formar um cluster denso (geralmente 2*dimensões ou min 5).
dbscan = DBSCAN(eps=5, min_samples=56)

# 2. Ajustar e prever (O DBSCAN não tem 'predict' separado para novos dados, ele ajusta o que tem)
labels = dbscan.fit_predict(X_treino)

# 3. Adicionar os labels ao seu DataFrame original para análise
df_resultado = pd.DataFrame(X_treino) # ou use seu df original
df_resultado['Cluster_DBSCAN'] = labels

# 4. Contagem dos Clusters
# O Cluster -1 é o ruído (Outlier/Possível Fraude)
print(df_resultado['Cluster_DBSCAN'].value_counts())
