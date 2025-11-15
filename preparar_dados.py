import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

print("Preparação e Normalização...")

try:
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
except FileNotFoundError:
    print("ERRO: Arquivo 'heart_failure_clinical_records_dataset.csv' não encontrado.")
    print("Certifique-se que ele está na mesma pasta do script.")
    exit()

print("Dados carregados com sucesso.")

features = df.drop(columns=['time', 'DEATH_EVENT'])

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

df_scaled = pd.DataFrame(features_scaled, columns=features.columns)

df_scaled.to_csv("heart_failure_scaled.csv", index=False)
joblib.dump(scaler, 'scaler.joblib')
