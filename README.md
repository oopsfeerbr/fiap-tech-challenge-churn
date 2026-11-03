# 📊 Previsão de Churn de Clientes — FIAP Tech Challenge

## 1. Sobre o projeto

Este projeto foi desenvolvido como parte do **Tech Challenge — Fase 1** da Pós-Tech em **Machine Learning Engineering da FIAP**.

O objetivo é desenvolver um pipeline de Machine Learning capaz de prever a probabilidade de cancelamento de clientes (*Customer Churn*) de uma empresa de telecomunicações.

A solução contempla:

* Entendimento do problema de negócio.
* Análise exploratória dos dados (EDA).
* Preparação e tratamento dos dados.
* Treinamento de modelos de classificação.
* Comparação de algoritmos utilizando validação cruzada.
* Seleção e persistência do modelo campeão.
* Disponibilização de previsões por API REST.
* Testes automatizados.
* Documentação técnica e de negócio.

O projeto utiliza Python, Scikit-Learn e FastAPI, com código versionado no GitHub.

---

## 2. Objetivo de negócio

O cancelamento de clientes representa um desafio para empresas de telecomunicações, pois pode provocar perda de receita recorrente e aumentar a necessidade de aquisição de novos consumidores.

O projeto busca identificar clientes com maior probabilidade de cancelamento, oferecendo informações que possam apoiar estratégias preventivas de retenção.

Entre as possíveis aplicações estão:

* Identificação de clientes com maior risco de churn.
* Priorização de campanhas de retenção.
* Apoio a estratégias de relacionamento.
* Melhor direcionamento de recursos comerciais.
* Identificação de segmentos que merecem atenção.

**Problema de Machine Learning:** classificação binária supervisionada.

**Variável-alvo:** `Churn Label`.

| Classe | Significado              |
| ------ | ------------------------ |
| 0      | Cliente que não cancelou |
| 1      | Cliente que cancelou     |

Os benefícios empresariais representam aplicações potenciais da solução. O projeto não realizou campanhas reais de retenção nem comprovou redução financeira de churn.

O planejamento do problema de negócio, os stakeholders e as métricas empresariais estão documentados no [ML Canvas](docs/ML_CANVAS.md).

---

## 3. Dataset

Foi utilizado o dataset público **IBM Telco Customer Churn**, disponibilizado por meio do Kaggle.

### Características do conjunto original

| Indicador                       | Resultado |
| ------------------------------- | --------: |
| Quantidade de registros         |     7.043 |
| Quantidade original de colunas  |        33 |
| Clientes que permaneceram       |     5.174 |
| Clientes que cancelaram         |     1.869 |
| Taxa de churn                   |    26,54% |
| Taxa de permanência             |    73,46% |
| Variáveis preditoras utilizadas |        20 |

O conjunto de dados contém informações demográficas, contratuais, financeiras e relacionadas aos serviços contratados pelos clientes.

### 3.1. Como obter o dataset

Para reproduzir as análises e o treinamento:

1. Acesse o Kaggle e localize o dataset **IBM Telco Customer Churn**.
2. Baixe a versão que contém o arquivo `Telco_customer_churn.xlsx`.
3. Coloque o arquivo Excel na pasta principal do projeto, no mesmo nível do `README.md`.
4. Mantenha exatamente o nome `Telco_customer_churn.xlsx`.

A localização esperada é:

```text
fiap-tech-challenge-churn/
├── Telco_customer_churn.xlsx
├── README.md
├── requirements.txt
└── ...
```

**Importante:** o dataset original não está incluído no repositório GitHub.

Ele deve ser obtido separadamente para reproduzir as análises e o treinamento.

A API pode ser executada sem o dataset original, pois utiliza o modelo previamente treinado e armazenado em `models/churn_model.joblib`.

---

## 4. Análise exploratória dos dados (EDA)

A análise exploratória teve como objetivo compreender a distribuição das variáveis e identificar características associadas ao cancelamento.

O notebook principal está disponível em:

`notebooks/analise_exploratoria.ipynb`

Também foi desenvolvido o script:

`analise_eda.py`

### 4.1. Principais descobertas

#### Tipo de contrato

| Tipo de contrato | Taxa de churn |
| ---------------- | ------------: |
| Mensal           |        42,71% |
| Um ano           |        11,27% |
| Dois anos        |         2,83% |

Clientes com contratos mensais apresentaram maior proporção de cancelamentos.

#### Método de pagamento

Clientes que utilizam **Electronic Check** apresentaram taxa de churn de **45,29%**.

#### Serviço de internet

Clientes com fibra óptica apresentaram taxa de churn de **41,89%**.

#### Tempo de permanência

| Grupo                     | Tempo médio |
| ------------------------- | ----------: |
| Clientes que permaneceram | 37,57 meses |
| Clientes que cancelaram   | 17,98 meses |

Clientes que cancelaram apresentaram menor tempo médio de permanência.

#### Cobrança mensal

| Grupo                     | Valor médio |
| ------------------------- | ----------: |
| Clientes que permaneceram |   US$ 61,27 |
| Clientes que cancelaram   |   US$ 74,44 |

Clientes que cancelaram apresentaram cobrança mensal média superior.

**Observação:** os resultados representam associações observadas no dataset, não necessariamente relações de causa e efeito.

---

## 5. Preparação dos dados

As principais etapas de preparação foram:

1. Identificação de valores ausentes e inconsistências.
2. Conversão da variável `Total Charges` para formato numérico.
3. Tratamento de 11 registros com valores inválidos nessa variável.
4. Remoção de identificadores e colunas inadequadas para a modelagem.
5. Exclusão de variáveis com risco de vazamento de informações (*data leakage*).
6. Separação entre variáveis numéricas e categóricas.
7. Aplicação de transformações por meio de pipelines do Scikit-Learn.
8. Divisão dos dados em treinamento e teste.

### 5.1. Prevenção de vazamento de informações

Foram excluídas variáveis que poderiam revelar diretamente o cancelamento ou conter informações inadequadas para uma previsão antecipada.

Entre elas:

* `Churn Value`.
* `Churn Score`.
* `Churn Reason`.

### 5.2. Divisão dos dados

| Conjunto    | Proporção | Registros |
| ----------- | --------: | --------: |
| Treinamento |       80% |     5.634 |
| Teste       |       20% |     1.409 |

A divisão foi realizada com estratificação da variável-alvo e seed fixa.

O pré-processamento foi integrado ao pipeline para aplicar as transformações de maneira consistente durante o treinamento e a inferência.

---

## 6. Modelagem de Machine Learning

Foram desenvolvidos e comparados três algoritmos de classificação:

* **Logistic Regression:** modelo linear utilizado como baseline.
* **Random Forest:** algoritmo baseado em conjuntos de árvores de decisão.
* **MLPClassifier:** rede neural artificial do tipo Multilayer Perceptron.

A avaliação utilizou validação cruzada estratificada e um conjunto de teste separado.

### 6.1. Resultados da validação cruzada

| Modelo              | Accuracy | Precision | Recall |   F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | ---------: | ------: |
| Logistic Regression |   0,8120 |    0,6668 | 0,5826 | **0,6213** |  0,8588 |
| Random Forest       |   0,8009 |    0,6530 | 0,5338 |     0,5873 |  0,8455 |
| MLPClassifier       |   0,8097 |    0,6701 | 0,5585 |     0,6082 |  0,8563 |

### 6.2. Seleção do modelo campeão

O modelo selecionado foi a **Logistic Regression**.

A escolha considerou principalmente o **F1-Score médio na validação cruzada**, métrica que equilibra Precision e Recall.

A Regressão Logística apresentou o maior F1-Score médio entre os algoritmos avaliados.

### 6.3. Resultados no conjunto de teste

| Métrica   | Resultado |
| --------- | --------: |
| Accuracy  |    80,20% |
| Precision |    64,26% |
| Recall    |    57,22% |
| F1-Score  |    60,54% |
| ROC-AUC   |    84,89% |

### 6.4. Matriz de confusão

|                 | Previsto: Não Churn | Previsto: Churn |
| --------------- | ------------------: | --------------: |
| Real: Não Churn |                 916 |             119 |
| Real: Churn     |                 160 |             214 |

O modelo identificou corretamente **214 clientes que cancelaram**.

Entretanto, deixou de identificar **160 clientes com churn**, demonstrando oportunidades de melhoria na identificação de clientes em risco.

O Recall de 57,22% indica que aproximadamente 57 em cada 100 clientes que efetivamente cancelaram foram identificados pelo modelo, considerando o limiar de classificação utilizado.

---

## 7. Reprodutibilidade dos experimentos

O projeto utiliza configurações destinadas a reduzir a variabilidade decorrente de operações pseudoaleatórias.

| Configuração                        | Valor             |
| ----------------------------------- | ----------------- |
| Seed principal                      | `42`              |
| Treinamento                         | 80%               |
| Teste                               | 20%               |
| Estratificação treino/teste         | Sim               |
| Validação cruzada                   | `StratifiedKFold` |
| Quantidade de divisões              | 5                 |
| Embaralhamento na validação cruzada | Sim               |
| Seed da validação cruzada           | `42`              |

A seed é aplicada na divisão dos dados, na configuração dos algoritmos e na validação cruzada.

As dependências do projeto estão registradas no arquivo `requirements.txt`.

**Observação:** seeds fixadas favorecem a reprodutibilidade, mas não garantem resultados absolutamente idênticos entre diferentes sistemas operacionais, ambientes e versões de bibliotecas.

---

## 8. Persistência do modelo

O modelo campeão foi serializado utilizando Joblib.

**Arquivo:**

`models/churn_model.joblib`

O artefato permite reutilizar o pipeline treinado para realizar previsões sem executar novamente todo o processo de treinamento.

O script responsável pela persistência é:

`salvar_modelo.py`

---

## 9. Instalação e configuração do ambiente

### 9.1. Pré-requisitos

* Python instalado.
* Git, caso deseje clonar o repositório.
* Acesso ao terminal PowerShell no Windows.

O projeto foi desenvolvido e testado em ambiente com **Python 3.15**.

A compatibilidade com outras versões de Python não foi validada neste projeto.

### 9.2. Obter o projeto

Clone o repositório:

```powershell
git clone https://github.com/oopsfeerbr/fiap-tech-challenge-churn.git
```

Acesse a pasta:

```powershell
cd fiap-tech-challenge-churn
```

### 9.3. Criar o ambiente virtual

```powershell
python -m venv .venv
```

### 9.4. Ativar o ambiente virtual

No PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Caso a política de execução do PowerShell impeça a ativação, é possível utilizar diretamente o interpretador localizado em `.venv\Scripts\python.exe`.

### 9.5. Instalar as dependências

Com o ambiente virtual ativado:

```powershell
python -m pip install -r requirements.txt
```

As dependências necessárias estão documentadas no arquivo `requirements.txt`.

---

## 10. Reprodução das análises e do treinamento

Para reproduzir os experimentos, primeiro obtenha o dataset `Telco_customer_churn.xlsx` e coloque-o na pasta principal do projeto.

Com o ambiente configurado, os scripts podem ser executados na seguinte ordem sugerida.

### 10.1. Análise exploratória

```powershell
python analise_eda.py
```

O projeto também disponibiliza a análise exploratória em formato Jupyter Notebook:

`notebooks/analise_exploratoria.ipynb`

### 10.2. Preparação dos dados

```powershell
python preparacao_dados.py
```

### 10.3. Treinamento do baseline

```powershell
python treinamento_baseline.py
```

### 10.4. Comparação dos algoritmos

```powershell
python comparacao_modelos.py
```

Esse script gera o arquivo:

`resultados/comparacao_modelos.csv`

### 10.5. Treinamento e persistência do modelo

```powershell
python salvar_modelo.py
```

Esse script salva o modelo treinado em:

`models/churn_model.joblib`

Os scripts utilizam o dataset original localizado na raiz do projeto.

**Importante:** a sequência apresentada representa um fluxo sugerido para reproduzir as etapas do projeto. Os scripts principais carregam diretamente o arquivo Excel, sem depender necessariamente da execução prévia dos demais.

A API não exige que o treinamento seja repetido, pois o artefato do modelo já está incluído no repositório.

---

## 11. API REST

Foi desenvolvida uma API REST utilizando FastAPI para disponibilizar as previsões do modelo.

### 11.1. Endpoints

| Método | Endpoint   | Descrição                         |
| ------ | ---------- | --------------------------------- |
| GET    | `/`        | Informações iniciais da aplicação |
| GET    | `/health`  | Verificação de funcionamento      |
| POST   | `/predict` | Previsão de churn                 |

### 11.2. Executar a API

Na pasta principal do projeto, execute:

```powershell
python -m uvicorn src.main:app --host 127.0.0.1 --port 8001
```

A API será disponibilizada localmente na porta `8001`.

### 11.3. Documentação interativa

Com a API em execução, acesse:

`http://127.0.0.1:8001/docs`

O Swagger UI permite visualizar os endpoints, consultar os esquemas de entrada e enviar requisições de teste.

### 11.4. Exemplo de resposta

```json
{
  "previsao_churn": 1,
  "probabilidade_churn": 0.7893,
  "classificacao": "Churn"
}
```

Nesse exemplo, o modelo classificou o cliente como potencial cancelamento, com probabilidade estimada de 78,93%.

O valor representa uma estimativa estatística, não uma certeza de cancelamento.

---

## 12. Testes automatizados

Foram implementados testes utilizando Pytest e o cliente de testes do FastAPI.

Os testes verificam:

1. Funcionamento do endpoint `/health`.
2. Previsão com dados válidos.
3. Rejeição de requisições com dados incompletos.
4. Rejeição de requisições com dados numéricos inválidos.

### 12.1. Executar os testes

Na pasta principal do projeto:

```powershell
python -m pytest tests/test_api.py -v
```

### 12.2. Resultado obtido

**4 testes aprovados.**

Os testes verificam comportamentos específicos da aplicação e ajudam a identificar regressões em funcionalidades essenciais da API.

---

## 13. Estrutura do projeto

```text
fiap-tech-challenge-churn/
│
├── data/
│   └── .gitkeep
│
├── docs/
│   ├── ML_CANVAS.md
│   └── model_card.md
│
├── models/
│   └── churn_model.joblib
│
├── notebooks/
│   ├── .gitkeep
│   └── analise_exploratoria.ipynb
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
├── .gitignore
├── Dataset - Kaggle (IBM).py
├── README.md
├── analise_eda.py
├── comparacao_modelos.py
├── preparacao_dados.py
├── requirements.txt
├── salvar_modelo.py
└── treinamento_baseline.py
```

### Responsabilidade dos principais componentes

| Arquivo ou diretório      | Finalidade                                   |
| ------------------------- | -------------------------------------------- |
| `docs/`                   | Documentação técnica e de negócio            |
| `models/`                 | Armazenamento do modelo treinado             |
| `notebooks/`              | Análise exploratória em Jupyter Notebook     |
| `resultados/`             | Resultados dos experimentos                  |
| `src/`                    | Implementação da API e da inferência         |
| `tests/`                  | Testes automatizados                         |
| `analise_eda.py`          | Análise exploratória                         |
| `preparacao_dados.py`     | Preparação e análise dos dados               |
| `treinamento_baseline.py` | Treinamento do modelo baseline               |
| `comparacao_modelos.py`   | Comparação dos algoritmos                    |
| `salvar_modelo.py`        | Treinamento e persistência do modelo campeão |
| `requirements.txt`        | Dependências do ambiente                     |

---

## 14. Tecnologias utilizadas

* Python.
* Pandas.
* NumPy.
* Scikit-Learn.
* Matplotlib.
* Joblib.
* FastAPI.
* Uvicorn.
* Pydantic.
* Pytest.
* HTTPX.
* Jupyter Notebook.

---

## 15. Documentação complementar

O projeto disponibiliza dois documentos complementares.

### 15.1. ML Canvas

**Arquivo:** [docs/ML_CANVAS.md](docs/ML_CANVAS.md)

Apresenta:

* Problema de negócio.
* Objetivos da solução.
* Stakeholders.
* Fonte de dados.
* Variável-alvo.
* Abordagem de Machine Learning.
* Métricas técnicas.
* Métricas de negócio.
* Riscos e limitações.
* Critérios de sucesso.

### 15.2. Model Card

**Arquivo:** [docs/model_card.md](docs/model_card.md)

Documenta:

* Identificação e finalidade do modelo.
* Dataset e variáveis utilizadas.
* Pré-processamento.
* Algoritmos avaliados.
* Critérios de seleção do modelo campeão.
* Métricas e matriz de confusão.
* Disponibilização por API.
* Testes automatizados.
* Limitações conhecidas.
* Possíveis vieses.
* Reprodutibilidade.

---

## 16. Limitações

O projeto apresenta algumas limitações importantes:

* O treinamento utiliza um dataset público específico do setor de telecomunicações.
* O desempenho em dados de outras empresas ou períodos pode diferir dos resultados observados.
* O modelo não identifica todos os clientes que efetivamente cancelam.
* As probabilidades produzidas não garantem o comportamento futuro dos clientes.
* Não foi realizada uma avaliação específica de calibração probabilística.
* Não foi conduzida uma auditoria quantitativa de equidade entre grupos demográficos.
* Não foi demonstrado que todas as variáveis estariam disponíveis antes do cancelamento em uma operação real.
* A solução ainda não contempla monitoramento contínuo de desempenho em produção.

O modelo deve ser utilizado como ferramenta de apoio à decisão, e não como única fonte para decisões comerciais de alto impacto.

---

## 17. Melhorias futuras

Entre as possibilidades de evolução estão:

* Otimização de hiperparâmetros.
* Ajuste do limiar de classificação para priorizar Recall.
* Avaliação de técnicas de tratamento do desbalanceamento.
* Análise de importância das variáveis.
* Explicabilidade das previsões.
* Avaliação da calibração das probabilidades.
* Monitoramento de desempenho e mudanças na distribuição dos dados.
* Avaliação de possíveis diferenças de desempenho entre grupos de clientes.
* Validação com dados mais recentes e representativos.
* Implantação da API em ambiente de nuvem.

---

## 18. Conclusão

O projeto demonstrou a aplicação de um fluxo de Machine Learning para previsão de churn, desde o entendimento do problema de negócio até a disponibilização de um modelo preditivo por meio de uma API REST.

Foram realizadas análise exploratória, preparação dos dados, comparação de algoritmos, validação cruzada, seleção do modelo campeão e persistência do pipeline treinado.

A **Logistic Regression** apresentou o maior F1-Score médio na validação cruzada, sendo selecionada como modelo final.

A API foi validada por meio de requisições de previsão e quatro testes automatizados aprovados.

A solução constitui uma prova de conceito acadêmica que poderá servir como base para futuras aplicações de apoio à retenção de clientes.
