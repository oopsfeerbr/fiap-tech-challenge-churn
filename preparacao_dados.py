from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_DATASET = PASTA_PROJETO / "Telco_customer_churn.xlsx"


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

df = pd.read_excel(CAMINHO_DATASET)

print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas.")


# ============================================================
# 3. TRATAMENTO DE TOTAL CHARGES
# ============================================================

# Converte Total Charges para número.
# Valores vazios são transformados em NaN.
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

# Foram encontrados clientes com Tenure Months = 0
# e Total Charges vazio.
# Como ainda não possuem tempo acumulado de contrato,
# tratamos Total Charges como 0.
df["Total Charges"] = df["Total Charges"].fillna(0)


# ============================================================
# 4. REMOÇÃO DE COLUNAS SEM PODER PREDITIVO
# ============================================================

colunas_sem_informacao = [
    "Count",
    "Country",
    "State"
]

df = df.drop(columns=colunas_sem_informacao)


# ============================================================
# 5. REMOÇÃO DE IDENTIFICADORES
# ============================================================

# CustomerID identifica individualmente cada cliente,
# mas não representa uma característica preditiva.
df = df.drop(columns=["CustomerID"])


# ============================================================
# 6. REMOÇÃO DE VARIÁVEIS COM DATA LEAKAGE
# ============================================================

# Essas variáveis possuem informação relacionada diretamente
# ao churn ocorrido e não devem ser utilizadas para prever
# churn futuro.
colunas_leakage = [
    "Churn Value",
    "Churn Score",
    "Churn Reason"
]

df = df.drop(columns=colunas_leakage)


# ============================================================
# 7. SEPARAÇÃO ENTRE FEATURES E TARGET
# ============================================================

X = df.drop(columns=["Churn Label"])

y = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 8. VALIDAÇÕES
# ============================================================

print("\n--- BASE TRATADA ---")

print(f"Linhas: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")

print(f"\nFeatures disponíveis: {X.shape[1]}")
print(f"Target disponível: {y.name}")

print("\nDistribuição do target:")
print(y.value_counts())

print("\nValores nulos:")
print(df.isnull().sum().sum())

print("\nTipo de Total Charges:")
print(df["Total Charges"].dtype)

print("\nPreparação inicial concluída com sucesso.")