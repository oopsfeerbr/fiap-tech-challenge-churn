from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_DATASET = PASTA_PROJETO / "Telco_customer_churn.xlsx"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

df = pd.read_excel(CAMINHO_DATASET)

print("\n============================================================")
print("TREINAMENTO DO BASELINE - REGRESSÃO LOGÍSTICA")
print("============================================================")

print(
    f"\nDataset carregado: "
    f"{df.shape[0]} linhas e {df.shape[1]} colunas."
)


# ============================================================
# 3. TRATAMENTO DE TOTAL CHARGES
# ============================================================

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

# Na EDA identificamos 11 clientes com Tenure Months = 0.
# Por isso, Total Charges vazio será tratado como 0.
df["Total Charges"] = df["Total Charges"].fillna(0)


# ============================================================
# 4. REMOÇÃO DE COLUNAS NÃO UTILIZADAS
# ============================================================

colunas_remover = [
    # Identificador
    "CustomerID",

    # Colunas constantes
    "Count",
    "Country",
    "State",

    # Variáveis com risco de data leakage
    "Churn Value",
    "Churn Score",
    "Churn Reason",

    # Variáveis geográficas removidas do baseline
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude"
]

df = df.drop(columns=colunas_remover)


# ============================================================
# 5. DEFINIÇÃO DAS FEATURES (X) E TARGET (y)
# ============================================================

X = df.drop(columns=["Churn Label"])

y = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})

print("\n--- DADOS PARA MODELAGEM ---")
print(f"Quantidade de registros: {len(X)}")
print(f"Quantidade de features: {X.shape[1]}")

print("\nDistribuição do target:")
print(y.value_counts())

print("\nDistribuição percentual:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# 6. IDENTIFICAÇÃO DAS COLUNAS NUMÉRICAS E CATEGÓRICAS
# ============================================================

colunas_numericas = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

colunas_categoricas = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\n--- FEATURES NUMÉRICAS ---")
print(colunas_numericas)

print("\n--- FEATURES CATEGÓRICAS ---")
print(colunas_categoricas)


# ============================================================
# 7. DIVISÃO ENTRE TREINO E TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n--- DIVISÃO TREINO / TESTE ---")
print(f"Treino: {len(X_train)} registros")
print(f"Teste:  {len(X_test)} registros")


# ============================================================
# 8. PREPROCESSAMENTO
# ============================================================

# Variáveis numéricas:
# StandardScaler coloca as features em escalas comparáveis.
preprocessamento_numerico = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

# Variáveis categóricas:
# OneHotEncoder transforma categorias em representação numérica.
preprocessamento_categorico = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# ColumnTransformer aplica tratamentos diferentes
# dependendo do tipo da variável.
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
# 9. CRIAÇÃO DO PIPELINE DE MACHINE LEARNING
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
# 10. TREINAMENTO
# ============================================================

print("\nTreinando Regressão Logística...")

modelo.fit(
    X_train,
    y_train
)

print("Treinamento concluído!")


# ============================================================
# 11. PREVISÕES NO CONJUNTO DE TESTE
# ============================================================

y_pred = modelo.predict(X_test)

# Probabilidade da classe positiva (churn = 1)
y_proba = modelo.predict_proba(X_test)[:, 1]


# ============================================================
# 12. MÉTRICAS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n============================================================")
print("RESULTADOS DO BASELINE")
print("============================================================")

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ============================================================
# 13. MATRIZ DE CONFUSÃO
# ============================================================

matriz = confusion_matrix(
    y_test,
    y_pred
)

print("\n--- MATRIZ DE CONFUSÃO ---")
print(matriz)

print("\nFormato:")
print("[[Verdadeiro Negativo, Falso Positivo],")
print(" [Falso Negativo, Verdadeiro Positivo]]")


# ============================================================
# 14. RELATÓRIO DE CLASSIFICAÇÃO
# ============================================================

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Não Churn",
            "Churn"
        ]
    )
)


# ============================================================
# 15. FINALIZAÇÃO
# ============================================================

print("============================================================")
print("BASELINE CONCLUÍDO COM SUCESSO")
print("============================================================")