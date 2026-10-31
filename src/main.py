from fastapi import FastAPI

from src.schemas import ClienteChurn
from src.predict import prever_churn


# ============================================================
# 1. CONFIGURAÇÃO DA API
# ============================================================

app = FastAPI(
    title="FIAP - API de Previsão de Churn",
    description="API REST para previsão de churn de clientes.",
    version="1.0.0"
)


# ============================================================
# 2. ENDPOINT PRINCIPAL
# ============================================================

@app.get("/")
def home():
    return {
        "projeto": "FIAP - Tech Challenge",
        "aplicacao": "API de Previsão de Churn",
        "status": "online"
    }


# ============================================================
# 3. ENDPOINT HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo": "carregado",
        "mensagem": "API de Churn funcionando corretamente!"
    }


# ============================================================
# 4. ENDPOINT DE PREVISÃO
# ============================================================

@app.post("/predict")
def predict(cliente: ClienteChurn):

    dados = cliente.model_dump(by_alias=True)

    resultado = prever_churn(dados)

    return resultado