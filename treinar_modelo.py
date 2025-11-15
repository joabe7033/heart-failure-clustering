import pandas as pd
from sklearn.cluster import KMeans
import joblib

OPTIMAL_K = 4

print(f"Treinamento final com K={OPTIMAL_K}...")

try:
    df_scaled = pd.read_csv("heart_failure_scaled.csv")
except FileNotFoundError:
    print("ERRO: Arquivo 'heart_failure_scaled.csv' não encontrado.")
    print("Rode o script '1_preparar_dados.py' primeiro.")
    exit()

print("Dados normalizados carregados com sucesso.")

print("Treinando o modelo K-Means...")
final_model = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
final_model.fit(df_scaled)

joblib.dump(final_model, 'heart_failure_kmeans.model')

print("-" * 30)
print(f"O 'cérebro' do modelo (K=4) foi salvo como 'heart_failure_kmeans.model'.")
print("-" * 30)