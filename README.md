# 📊 Previsão de Churn de Clientes — FIAP Tech Challenge

## 1. Sobre o projeto

Este projeto foi desenvolvido como parte do Tech Challenge da Pós-Tech em Machine Learning Engineering da FIAP.

O objetivo é construir um pipeline de Machine Learning capaz de prever a probabilidade de cancelamento de clientes (*Customer Churn*) de uma empresa de telecomunicações.

A solução contempla análise exploratória, preparação dos dados, treinamento e comparação de modelos, disponibilização de previsões por API REST e testes automatizados.

## 2. Objetivo de negócio

O cancelamento de clientes representa um desafio para empresas de telecomunicações, pois afeta diretamente a receita e aumenta a necessidade de aquisição de novos consumidores.

O projeto busca identificar clientes com maior probabilidade de cancelamento, permitindo que a empresa desenvolva estratégias preventivas de retenção.

**Problema de Machine Learning:** classificação binária.

**Variável-alvo:** `Churn Label`.

* `0`: cliente que não cancelou.
* `1`: cliente que cancelou.

## 3. Dataset

Foi utilizado o dataset **Telco Customer Churn**, disponibilizado pela IBM e obtido por meio do Kaggle.

Características do conjunto original:

| Indicador                      | Resultado |
| ------------------------------ | --------- |
| Quantidade de registros        | 7.043     |
| Quantidade original de colunas | 33        |
| Clientes que permaneceram      | 5.174     |
| Clientes que cancelaram        | 1.869     |
| Taxa de churn                  | 26,54%    |
| Taxa de permanência            | 73,46%    |

Para a modelagem, foram selecionadas 20 variáveis preditoras.

### Como obter o dataset

O projeto utiliza o dataset público Telco Customer Churn,
disponibilizado pela IBM e acessível por meio do Kaggle.

Para reproduzir as análises e o treinamento:

1. Acesse o Kaggle e procure pelo dataset IBM Telco Customer Churn.
2. Baixe a versão que contém o arquivo Telco_customer_churn.xlsx.
3. Coloque o arquivo na pasta principal do projeto, no mesmo nível do README.md.
4. Mantenha exatamente o nome Telco_customer_churn.xlsx.

Estrutura esperada:

fiap-tech-challenge-churn/
├── Telco_customer_churn.xlsx
├── README.md
├── requirements.txt
├── analise_eda.py
├── preparacao_dados.py
└── salvar_modelo.py

Observação: o dataset original não está incluído no repositório GitHub.
O arquivo deve ser obtido separadamente para executar os scripts de
análise exploratória, preparação e treinamento.

A API de previsão pode ser executada sem o dataset original,
pois utiliza o modelo previamente treinado e salvo em
models/churn_model.joblib.

## 4. Análise exploratória dos dados (EDA)

A análise exploratória permitiu compreender o comportamento dos clientes e identificar características associadas ao cancelamento.

### Principais descobertas

**Tipo de contrato**

* Contratos mensais: 42,71% de churn.
* Contratos anuais: 11,27% de churn.
* Contratos de dois anos: 2,83% de churn.

**Método de pagamento**

Clientes que utilizam *Electronic Check* apresentam uma taxa de churn de 45,29%.

**Serviço de internet**

Clientes com fibra óptica apresentam taxa de churn de 41,89%.

**Tempo de permanência**

* Clientes que permaneceram: média de 37,57 meses.
* Clientes que cancelaram: média de 17,98 meses.

**Cobrança mensal**

* Clientes que permaneceram: média de US$ 61,27.
* Clientes que cancelaram: média de US$ 74,44.

Esses resultados representam associações observadas nos dados, não necessariamente relações de causa e efeito.

## 5. Preparação dos dados

As principais etapas de preparação foram:

1. Identificação de valores ausentes e inconsistências.
2. Conversão da variável `Total Charges` para formato numérico.
3. Tratamento de 11 registros com valores inválidos nessa variável.
4. Remoção de colunas sem utilidade preditiva ou com risco de vazamento de informação.
5. Separação entre variáveis numéricas e categóricas.
6. Aplicação de transformações apropriadas às variáveis.
7. Divisão dos dados em treinamento (80%) e teste (20%).

O processamento das variáveis foi integrado ao pipeline de modelagem utilizando Scikit-Learn.

## 6. Modelos de Machine Learning

Foram desenvolvidos e comparados três algoritmos:

* Logistic Regression.
* Random Forest.
* MLPClassifier (rede neural).

Os modelos foram avaliados utilizando validação cruzada e métricas de classificação.

### Resultados da validação cruzada

| Modelo              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0,8120 |    0,6668 | 0,5826 |   0,6213 |  0,8588 |
| Random Forest       |   0,8009 |    0,6530 | 0,5338 |   0,5873 |  0,8455 |
| MLPClassifier       |   0,8097 |    0,6701 | 0,5585 |   0,6082 |  0,8563 |

### Modelo selecionado

**Logistic Regression**

A Regressão Logística foi selecionada por apresentar o melhor F1-Score médio na validação cruzada.

### Resultados no conjunto de teste

| Métrica   | Resultado |
| --------- | --------: |
| Accuracy  |    80,20% |
| Precision |    64,26% |
| Recall    |    57,22% |
| F1-Score  |    60,54% |
| ROC-AUC   |    84,89% |

O Recall indica que o modelo identificou aproximadamente 57% dos clientes que efetivamente cancelaram no conjunto de teste.

Esse resultado demonstra potencial de aplicação, mas também evidencia oportunidades de melhoria para identificar uma parcela maior dos clientes em risco.

## 7. Persistência do modelo

O modelo selecionado foi treinado e salvo utilizando a biblioteca Joblib.

Arquivo:

`models/churn_model.joblib`

O artefato permite reutilizar o pipeline treinado para realizar previsões sem necessidade de executar novamente todo o treinamento.

## 8. API REST

Foi desenvolvida uma API utilizando FastAPI para disponibilizar as previsões do modelo.

### Endpoints

| Método | Endpoint   | Descrição                    |
| ------ | ---------- | ---------------------------- |
| GET    | `/`        | Informações iniciais da API  |
| GET    | `/health`  | Verificação de funcionamento |
| POST   | `/predict` | Previsão de churn            |

### Executando a API

Na pasta principal do projeto, execute:

```powershell
python -m uvicorn src.main:app --host 127.0.0.1 --port 8001
```

Caso o comando `python` não utilize o ambiente correto, execute com o caminho do interpretador Python configurado no projeto.

### Documentação interativa

Com a API em execução, acesse:

`http://127.0.0.1:8001/docs`

O Swagger UI permite consultar os endpoints e enviar requisições de teste.

### Exemplo de resposta

```json
{
  "previsao_churn": 1,
  "probabilidade_churn": 0.7893,
  "classificacao": "Churn"
}
```

Nesse exemplo, o modelo classificou o cliente como potencial cancelamento, com probabilidade estimada de 78,93%.

## 9. Testes automatizados

Foram implementados testes utilizando Pytest e o TestClient do FastAPI.

Os testes verificam:

1. Funcionamento do endpoint `/health`.
2. Previsão com dados válidos.
3. Rejeição de requisições com dados incompletos.
4. Rejeição de requisições com dados inválidos.

### Executando os testes

```powershell
python -m pytest tests/test_api.py -v
```

**Resultado obtido: 4 testes aprovados.**

## 10. Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Joblib
* FastAPI
* Uvicorn
* Pydantic
* Pytest
* HTTPX

## 11. Instalação das dependências

Com Python instalado, execute na pasta principal:

```powershell
python -m pip install -r requirements.txt
```

As dependências utilizadas pelo projeto estão documentadas no arquivo `requirements.txt`.

## 12. Estrutura do projeto

```text
projeto-churn/
│
├── docs/
│   └── model_card.md
│
├── models/
│   └── churn_model.joblib
│
├── resultados/
│   └── comparacao_modelos.csv
│
├── src/
│   ├── main.py
│   ├── predict.py
│   └── schemas.py
│
├── tests/
│   └── test_api.py
│
├── analise_eda.py
├── comparacao_modelos.py
├── preparacao_dados.py
├── salvar_modelo.py
├── treinamento_baseline.py
├── requirements.txt
└── README.md
```

## 13. Limitações

O projeto apresenta algumas limitações importantes:

* O treinamento utiliza um conjunto de dados específico do setor de telecomunicações.
* O desempenho em novos dados reais pode diferir dos resultados observados.
* O modelo não identifica todos os clientes que efetivamente cancelam.
* As probabilidades produzidas pelo modelo não garantem o comportamento futuro dos clientes.
* A solução ainda não contempla monitoramento contínuo de desempenho em produção.

## 14. Próximas melhorias

Entre as possibilidades de evolução estão:

* Otimização de hiperparâmetros.
* Ajuste do limiar de classificação para priorizar a identificação de churn.
* Análise de importância das variáveis.
* Monitoramento de desempenho e possíveis mudanças na distribuição dos dados.
* Implantação da API em ambiente de nuvem.

## 15. Documentação complementar

Informações adicionais sobre o modelo, suas métricas e limitações estão disponíveis em:

`docs/model_card.md`

## 16. Conclusão

O projeto demonstrou a aplicação de um fluxo de Machine Learning, desde a compreensão dos dados até a disponibilização de um modelo preditivo por meio de uma API REST.

A Regressão Logística apresentou o melhor desempenho médio em F1-Score durante a validação cruzada, sendo selecionada como modelo final.

A solução desenvolvida oferece uma base para identificação de clientes com risco de cancelamento e para futuras melhorias orientadas à retenção.
