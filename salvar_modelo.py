from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent

CAMINHO_DATASET = PASTA_PROJETO / "Telco_customer_churn.xlsx"

PASTA_MODELOS = PASTA_PROJETO / "models"
PASTA_MODELOS.mkdir(exist_ok=True)

CAMINHO_MODELO = PASTA_MODELOS / "churn_model.joblib"

RANDOM_STATE = 42


# ============================================================
# 2. CARREGAMENTO DO DATASET
# ============================================================

print("\n======================================================")
print("PROJETO FIAP - SALVANDO MODELO CAMPEÃO")
print("======================================================")

df = pd.read_excel(CAMINHO_DATASET)

print(f"\nDataset carregado: {df.shape}")


# ============================================================
# 3. TRATAMENTO DOS DADOS
# ============================================================

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

valores_invalidos = df["Total Charges"].isna()

if (
    df.loc[valores_invalidos, "Tenure Months"] != 0
).any():
    raise ValueError(
        "Total Charges possui valores inválidos "
        "para clientes com Tenure Months diferente de zero."
    )

df["Total Charges"] = df["Total Charges"].fillna(0)


# ============================================================
# 4. REMOÇÃO DE COLUNAS
# ============================================================

colunas_remover = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude"
]

df = df.drop(columns=colunas_remover)


# ============================================================
# 5. DEFINIÇÃO DE X E y
# ============================================================

X = df.drop(columns=["Churn Label"])

y = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Foram encontrados valores inválidos no target.")

print(f"\nQuantidade de features: {X.shape[1]}")


# ============================================================
# 6. DIVISÃO TREINO / TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# 7. IDENTIFICAÇÃO DAS FEATURES
# ============================================================

colunas_numericas = X_train.select_dtypes(
    include="number"
).columns.tolist()

colunas_categoricas = X_train.select_dtypes(
    include=["object", "str", "string", "category"]
).columns.tolist()


# ============================================================
# 8. PREPROCESSAMENTO
# ============================================================

preprocessamento_numerico = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

preprocessamento_categorico = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            preprocessamento_numerico,
            colunas_numericas
        ),
        (
            "categoricas",
            preprocessamento_categorico,
            colunas_categoricas
        )
    ]
)


# ============================================================
# 9. MODELO CAMPEÃO
# ============================================================

modelo = Pipeline(
    steps=[
        ("preprocessamento", preprocessador),

        (
            "modelo",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ============================================================
# 10. TREINAMENTO DO MODELO
# ============================================================

print("\nTreinando modelo campeão...")

modelo.fit(X_train, y_train)

print("Modelo treinado com sucesso!")


# ============================================================
# 11. SALVAMENTO DO MODELO
# ============================================================

joblib.dump(
    modelo,
    CAMINHO_MODELO
)

print(f"\nModelo salvo em: {CAMINHO_MODELO}")


# ============================================================
# 12. TESTE DE CARREGAMENTO
# ============================================================

print("\nTestando carregamento do modelo...")

modelo_carregado = joblib.load(CAMINHO_MODELO)

# Selecionamos um cliente do conjunto de teste.
cliente_exemplo = X_test.iloc[[0]]

previsao = modelo_carregado.predict(cliente_exemplo)[0]

probabilidade = modelo_carregado.predict_proba(
    cliente_exemplo
)[0][1]

print("\n--- PREVISÃO DE EXEMPLO ---")

print(f"Previsão: {previsao}")
print(f"Probabilidade de churn: {probabilidade:.2%}")

print("\n======================================================")
print("MODELO SALVO E VALIDADO COM SUCESSO!")
print("======================================================")