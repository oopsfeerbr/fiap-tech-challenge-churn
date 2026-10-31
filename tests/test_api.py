# ============================================================
# PROJETO FIAP - TECH CHALLENGE
# TESTES AUTOMATIZADOS DA API DE CHURN
#
# Biblioteca: Pytest
# API: FastAPI
# Modelo: Logistic Regression
# ============================================================

from fastapi.testclient import TestClient

from src.main import app


# ============================================================
# 1. CONFIGURAÇÃO DO CLIENTE DE TESTES
# ============================================================

client = TestClient(app)


# ============================================================
# 2. DADOS DE EXEMPLO PARA PREVISÃO
# ============================================================

cliente_valido = {
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 2,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 90.0,
    "Total Charges": 180.0,
    "CLTV": 3500
}


# ============================================================
# TESTE 1 - HEALTH CHECK
# ============================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    dados = response.json()

    assert dados["status"] == "ok"

    print("\nTESTE 1: Health Check aprovado!")


# ============================================================
# TESTE 2 - PREVISÃO COM DADOS VÁLIDOS
# ============================================================

def test_predict_valido():

    response = client.post(
        "/predict",
        json=cliente_valido
    )

    assert response.status_code == 200

    dados = response.json()

    assert "previsao_churn" in dados

    assert "probabilidade_churn" in dados

    assert "classificacao" in dados

    assert dados["previsao_churn"] in [0, 1]

    assert 0 <= dados["probabilidade_churn"] <= 1

    if dados["previsao_churn"] == 1:
        assert dados["classificacao"] == "Churn"
    else:
        assert dados["classificacao"] == "Não Churn"

    print("\nTESTE 2: Previsão válida aprovada!")


# ============================================================
# TESTE 3 - DADOS INCOMPLETOS
# ============================================================

def test_predict_dados_incompletos():

    cliente_incompleto = cliente_valido.copy()

    del cliente_incompleto["Contract"]

    response = client.post(
        "/predict",
        json=cliente_incompleto
    )

    assert response.status_code == 422

    print("\nTESTE 3: Validação de dados incompletos aprovada!")


# ============================================================
# TESTE 4 - DADOS INVÁLIDOS
# ============================================================

def test_predict_dados_invalidos():

    cliente_invalido = cliente_valido.copy()

    cliente_invalido["Tenure Months"] = -10

    response = client.post(
        "/predict",
        json=cliente_invalido
    )

    assert response.status_code == 422

    print("\nTESTE 4: Validação de dados inválidos aprovada!")