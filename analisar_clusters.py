import pandas as pd
import joblib

print ("Análise e Descrição dos Clusters...")

try:
    
    model = joblib.load('heart_failure_kmeans.model')
    df_scaled = pd.read_csv("heart_failure_scaled.csv")
    df_original = pd.read_csv("heart_failure_clinical_records_dataset.csv")
except FileNotFoundError:
    print("ERRO: Faltam arquivos. Rode os passos 1 e 3 primeiro.")
    exit()

labels = model.predict(df_scaled)

df_original['cluster'] = labels

cluster_description = df_original.groupby('cluster').mean().T

print("\n" + "="*40)
print("   MÉDIAS DOS GRUPOS (CLUSTERS K=4)   ")
print("="*40)
print(cluster_description)

cluster_description.to_csv("cluster_description.csv")
df_original.to_csv("resultado_final_com_clusters.csv", index=False)

print("\nArquivo 'resultado_final_clusters.csv' criado!")
print("Abra esse arquivo para ver qual paciente ficou em qual grupo.")