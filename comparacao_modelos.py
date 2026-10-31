from pathlib import Path
import warnings

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent

CAMINHO_DATASET = PASTA_PROJETO / "Telco_customer_churn.xlsx"

PASTA_RESULTADOS = PASTA_PROJETO / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

print("\n======================================================")
print("PROJETO FIAP - COMPARAÇÃO DE MODELOS DE CHURN")
print("======================================================")

df = pd.read_excel(CAMINHO_DATASET)

print(f"\nDataset original: {df.shape[0]} linhas e {df.shape[1]} colunas")


# ============================================================
# 3. TRATAMENTO DOS DADOS
# ============================================================

# Converte Total Charges para valores numéricos.
# Os valores vazios identificados na EDA pertencem
# a clientes com zero meses de permanência.

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

# Validamos a regra de negócio antes de preencher com zero.
valores_invalidos = df["Total Charges"].isna()

if (
    df.loc[valores_invalidos, "Tenure Months"] != 0
).any():
    raise ValueError(
        "Existem valores inválidos em Total Charges "
        "para clientes com Tenure Months diferente de zero."
    )

df["Total Charges"] = df["Total Charges"].fillna(0)


# ============================================================
# 4. REMOÇÃO DE COLUNAS
# ============================================================

colunas_remover = [

    # Identificador
    "CustomerID",

    # Colunas constantes
    "Count",
    "Country",
    "State",

    # Variáveis com vazamento de informação
    "Churn Value",
    "Churn Score",
    "Churn Reason",

    # Variáveis geográficas excluídas deste experimento
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude"
]

df = df.drop(columns=colunas_remover)

print(f"Dataset preparado: {df.shape[0]} linhas e {df.shape[1]} colunas")


# ============================================================
# 5. FEATURES E TARGET
# ============================================================

X = df.drop(columns=["Churn Label"])

y = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("O target possui valores inesperados.")

print("\n--- FEATURES E TARGET ---")
print(f"Quantidade de features: {X.shape[1]}")
print(f"Quantidade de registros: {len(X)}")

print("\nDistribuição do target:")
print(y.value_counts())


# ============================================================
# 6. IDENTIFICAÇÃO DOS TIPOS DE VARIÁVEIS
# ============================================================

colunas_numericas = X.select_dtypes(
    include="number"
).columns.tolist()

colunas_categoricas = X.select_dtypes(
    include=["object", "str", "string", "category"]
).columns.tolist()

print("\n--- VARIÁVEIS NUMÉRICAS ---")
print(colunas_numericas)

print("\n--- VARIÁVEIS CATEGÓRICAS ---")
print(colunas_categoricas)


# ============================================================
# 7. DIVISÃO TREINO E TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n--- DIVISÃO DOS DADOS ---")
print(f"Treinamento: {len(X_train)} registros")
print(f"Teste: {len(X_test)} registros")


# ============================================================
# 8. PREPROCESSAMENTO
# ============================================================

# Variáveis numéricas:
# 1. Preenchimento de valores ausentes
# 2. Padronização

preprocessamento_numerico = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# Variáveis categóricas:
# 1. Preenchimento de valores ausentes
# 2. One-Hot Encoding

preprocessamento_categorico = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
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
# 9. DEFINIÇÃO DOS MODELOS
# ============================================================

modelos = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "MLPClassifier": MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
}


# ============================================================
# 10. CONFIGURAÇÃO DA CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

metricas_cv = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ============================================================
# 11. TREINAMENTO E AVALIAÇÃO DOS MODELOS
# ============================================================

resultados = []

for nome_modelo, algoritmo in modelos.items():

    print("\n======================================================")
    print(f"MODELO: {nome_modelo}")
    print("======================================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            ("modelo", algoritmo)
        ]
    )

    # --------------------------------------------------------
    # CROSS-VALIDATION
    # --------------------------------------------------------

    print("\nExecutando Cross-Validation...")

    with warnings.catch_warnings():
        warnings.simplefilter("default", ConvergenceWarning)

        resultado_cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=metricas_cv,
            n_jobs=1,
            error_score="raise"
        )

    cv_accuracy = resultado_cv["test_accuracy"].mean()
    cv_precision = resultado_cv["test_precision"].mean()
    cv_recall = resultado_cv["test_recall"].mean()
    cv_f1 = resultado_cv["test_f1"].mean()
    cv_roc_auc = resultado_cv["test_roc_auc"].mean()

    print("\n--- RESULTADOS CROSS-VALIDATION ---")
    print(f"CV Accuracy:  {cv_accuracy:.4f}")
    print(f"CV Precision: {cv_precision:.4f}")
    print(f"CV Recall:    {cv_recall:.4f}")
    print(f"CV F1:        {cv_f1:.4f}")
    print(f"CV ROC-AUC:   {cv_roc_auc:.4f}")

    # --------------------------------------------------------
    # TREINAMENTO FINAL NO CONJUNTO DE TREINO
    # --------------------------------------------------------

    print("\nTreinando modelo com todos os dados de treinamento...")

    pipeline.fit(X_train, y_train)

    # --------------------------------------------------------
    # PREVISÕES NO CONJUNTO DE TESTE
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_test)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # MÉTRICAS DE TESTE
    # --------------------------------------------------------

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    test_recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    test_f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    test_roc_auc = roc_auc_score(y_test, y_proba)

    print("\n--- RESULTADOS NO TESTE ---")

    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_roc_auc:.4f}")

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    # --------------------------------------------------------
    # ARMAZENAMENTO DOS RESULTADOS
    # --------------------------------------------------------

    resultados.append({

        "Modelo": nome_modelo,

        "CV Accuracy": cv_accuracy,
        "CV Precision": cv_precision,
        "CV Recall": cv_recall,
        "CV F1": cv_f1,
        "CV ROC-AUC": cv_roc_auc,

        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1,
        "Test ROC-AUC": test_roc_auc

    })


# ============================================================
# 12. COMPARAÇÃO FINAL
# ============================================================

df_resultados = pd.DataFrame(resultados)

# Ordenação pelo F1 médio da validação cruzada.
df_resultados = df_resultados.sort_values(
    by="CV F1",
    ascending=False
)

print("\n======================================================")
print("COMPARAÇÃO FINAL DOS MODELOS")
print("======================================================\n")

print(
    df_resultados.round(4).to_string(index=False)
)


# ============================================================
# 13. MELHOR MODELO PELA CROSS-VALIDATION
# ============================================================

melhor_modelo = df_resultados.iloc[0]

print("\n======================================================")
print("MELHOR MODELO POR CV F1")
print("======================================================")

print(f"\nModelo: {melhor_modelo['Modelo']}")
print(f"CV F1: {melhor_modelo['CV F1']:.4f}")
print(f"CV ROC-AUC: {melhor_modelo['CV ROC-AUC']:.4f}")
print(f"Test F1: {melhor_modelo['Test F1']:.4f}")
print(f"Test ROC-AUC: {melhor_modelo['Test ROC-AUC']:.4f}")


# ============================================================
# 14. EXPORTAÇÃO DOS RESULTADOS
# ============================================================

CAMINHO_RESULTADOS = (
    PASTA_RESULTADOS / "comparacao_modelos.csv"
)

df_resultados.to_csv(
    CAMINHO_RESULTADOS,
    index=False,
    encoding="utf-8-sig"
)

print("\n======================================================")
print("COMPARAÇÃO CONCLUÍDA COM SUCESSO")
print("======================================================")

print(f"\nResultados salvos em: {CAMINHO_RESULTADOS}")