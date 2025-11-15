import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("Elbow Method...")

try:
    features_scaled = pd.read_csv("heart_failure_scaled.csv")
except FileNotFoundError:
    print("ERRO: 'heart_failure_scaled.csv' não encontrado.")
    print("Execute o '1_preparar_dados.py' primeiro.")
    exit()

print("Dados normalizados carregados.")

inertia = []
K = range(1, 11)

print("Calculando inércia para K de 1 a 10...")
for k in K:
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_model.fit(features_scaled)
    inertia.append(kmeans_model.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (WCSS)')
plt.title('Método do Cotovelo (Elbow Method)')
plt.grid(True)
plt.savefig("elbow_plot.png")
plt.show()