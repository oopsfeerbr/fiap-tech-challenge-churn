from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. LOCALIZAÇÃO DO MODELO
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_MODELO = (
    PASTA_PROJETO / "models" / "churn_model.joblib"
)


# ============================================================
# 2. CARREGAMENTO DO MODELO
# ============================================================

modelo = joblib.load(CAMINHO_MODELO)


# ============================================================
# 3. FUNÇÃO DE PREVISÃO
# ============================================================

def prever_churn(dados_cliente: dict):

    # Converte o JSON recebido em DataFrame
    cliente_df = pd.DataFrame([dados_cliente])

    # Executa a previsão
    previsao = int(modelo.predict(cliente_df)[0])

    # Calcula a probabilidade de churn
    probabilidade = float(
        modelo.predict_proba(cliente_df)[0][1]
    )

    return {
        "previsao_churn": previsao,
        "probabilidade_churn": round(probabilidade, 4),
        "classificacao": (
            "Churn"
            if previsao == 1
            else "Não Churn"
        )
    }