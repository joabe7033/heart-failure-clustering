import numpy as np
import pandas as pd
import joblib

print("Previsão de Novo Paciente...")

try:
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('heart_failure_kmeans.model')
except FileNotFoundError as e:
    print(f"ERRO CRÍTICO: Arquivo não encontrado: {e.filename}")
    print("Execute os scripts 1 e 3 antes deste.")
    exit()

print("Scaler e Modelo K-Means carregados.")

feature_names = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
    'ejection_fraction', 'high_blood_pressure', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking'
]

new_patient_data = [
    75.0,    # age
    0.0,     # anaemia
    582.0,   # creatinine_phosphokinase
    0.0,     # diabetes
    20.0,    # ejection_fraction
    1.0,     # high_blood_pressure
    265000.0,# platelets
    1.9,     # serum_creatinine
    130.0,   # serum_sodium
    1.0,     # sex
    0.0      # smoking
]

new_patient_df = pd.DataFrame([new_patient_data], columns=feature_names)

print("\n--- Dados do Novo Paciente (Não Normalizado) ---")
print(new_patient_df.T)

new_patient_scaled = scaler.transform(new_patient_df)

print("\nPaciente normalizado. Pronto para previsão...")

predicted_cluster = model.predict(new_patient_scaled)

# --- Dicionário com a análise K=4 ---
cluster_profiles = {
    0: "ALERTA DE ALTO RISCO (Perfil: Idosos com Falha Renal, 62% Mortalidade)",
    1: "BAIXO RISCO (Perfil: Homens Não-Fumantes, 24% Mortalidade)",
    2: "RISCO MÉDIO-BAIXO (Perfil: Mulheres c/ Alta Incidência de Diabetes/Pressão Alta, 29% Mortalidade)",
    3: "RISCO MÉDIO (Perfil: Homens Fumantes, 26% Mortalidade)"
}
# --------------------------------------------------

cluster_num = predicted_cluster[0]
result_description = cluster_profiles.get(predicted_cluster[0], "Cluster não identificado")

print("\n" + "="*40)
print("   RESULTADO DA PREVISÃO (TRADUZIDO)   ")
print("="*40)
print(f"Cluster numérico: {predicted_cluster[0]}")
print(f"Interpretação: {result_description}")
print("\nCONCLUÍDO!")