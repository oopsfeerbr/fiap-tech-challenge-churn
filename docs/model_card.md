# Model Card — Previsão de Churn de Clientes

**Projeto:** FIAP — Tech Challenge — Machine Learning Engineering
**Versão do modelo:** 1.0
**Algoritmo selecionado:** Logistic Regression
**Formato de armazenamento:** Joblib
**Arquivo:** `models/churn_model.joblib`

---

## 1. Visão geral

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de prever a probabilidade de cancelamento de serviços (churn) por clientes de uma empresa de telecomunicações.

A solução utiliza informações contratuais, demográficas e financeiras para identificar clientes com maior propensão ao cancelamento.

O modelo foi disponibilizado por meio de uma API REST desenvolvida com FastAPI, permitindo a realização de previsões a partir de requisições HTTP.

## 2. Objetivo de negócio

O principal objetivo é apoiar estratégias de retenção de clientes.

A identificação antecipada de clientes com maior risco de churn pode permitir que a empresa desenvolva ações preventivas, como ofertas personalizadas, melhorias no atendimento e campanhas de relacionamento.

A solução tem como possíveis benefícios:

* Identificação de clientes com maior risco de cancelamento.
* Priorização de campanhas de retenção.
* Apoio à tomada de decisões comerciais.
* Potencial redução da perda de clientes.
* Melhor direcionamento de recursos de relacionamento.

Esses benefícios representam aplicações potenciais e não resultados financeiros comprovados pelo projeto.

## 3. Dataset utilizado

**Nome:** IBM Telco Customer Churn
**Domínio:** Telecomunicações
**Tipo de problema:** Classificação binária

### Características do dataset

| Característica                    | Informação           |
| --------------------------------- | -------------------- |
| Quantidade de registros           | 7.043                |
| Quantidade original de colunas    | 33                   |
| Quantidade de features utilizadas | 20                   |
| Variável-alvo                     | Churn Label          |
| Classe 0                          | Cliente não cancelou |
| Classe 1                          | Cliente cancelou     |
| Registros sem churn               | 5.174                |
| Registros com churn               | 1.869                |

### Distribuição da variável-alvo

| Classe    | Quantidade | Percentual |
| --------- | ---------: | ---------: |
| Não Churn |      5.174 |     73,46% |
| Churn     |      1.869 |     26,54% |

O dataset apresenta desbalanceamento moderado entre as classes, com predominância de clientes que não cancelaram os serviços.

Por esse motivo, a avaliação considera métricas adicionais à acurácia, especialmente Precision, Recall, F1-Score e ROC-AUC.

## 4. Preparação dos dados

Durante a preparação dos dados, foram realizadas atividades como:

* Identificação de valores ausentes e inconsistentes.
* Conversão da variável `Total Charges` para formato numérico.
* Tratamento de valores inválidos.
* Remoção de identificadores e variáveis inadequadas para a modelagem.
* Separação das variáveis explicativas e da variável-alvo.
* Preparação de variáveis numéricas e categóricas.
* Divisão dos dados em conjuntos de treinamento e teste.

Foram identificados 11 registros com valores não numéricos em `Total Charges`, associados a clientes com zero meses de permanência.

Também foram identificadas variáveis que poderiam provocar vazamento de informações (*data leakage*), como:

* `Churn Value`: representação numérica direta da variável-alvo.
* `Churn Score`: pontuação previamente associada ao risco de churn.
* `Churn Reason`: informação relacionada ao motivo do cancelamento.

Essas variáveis foram excluídas das features utilizadas no treinamento.

O pré-processamento foi integrado ao pipeline do modelo para permitir a aplicação consistente das transformações durante a inferência.

## 5. Variáveis utilizadas

O modelo utiliza 20 características dos clientes.

### Variáveis numéricas

1. Tenure Months
2. Monthly Charges
3. Total Charges
4. CLTV

### Variáveis categóricas

1. Gender
2. Senior Citizen
3. Partner
4. Dependents
5. Phone Service
6. Multiple Lines
7. Internet Service
8. Online Security
9. Online Backup
10. Device Protection
11. Tech Support
12. Streaming TV
13. Streaming Movies
14. Contract
15. Paperless Billing
16. Payment Method

## 6. Algoritmos avaliados

Foram treinados e comparados três algoritmos de classificação:

**Logistic Regression:** modelo linear utilizado como baseline, com interpretação relativamente simples.

**Random Forest:** algoritmo baseado em um conjunto de árvores de decisão.

**MLPClassifier:** rede neural artificial do tipo Multilayer Perceptron.

A avaliação utilizou validação cruzada (*cross-validation*) e um conjunto de teste separado.

### Resultados da validação cruzada

| Modelo              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0,8120 |    0,6668 | 0,5826 |   0,6213 |  0,8588 |
| Random Forest       |   0,8009 |    0,6530 | 0,5338 |   0,5873 |  0,8455 |
| MLPClassifier       |   0,8097 |    0,6701 | 0,5585 |   0,6082 |  0,8563 |

A Logistic Regression apresentou o maior F1-Score médio na validação cruzada, sendo selecionada como modelo campeão.

## 7. Desempenho do modelo campeão

### Resultados no conjunto de teste

| Métrica   | Resultado |
| --------- | --------: |
| Accuracy  |    80,20% |
| Precision |    64,26% |
| Recall    |    57,22% |
| F1-Score  |    60,54% |
| ROC-AUC   |    84,89% |

### Matriz de confusão

|                 | Previsto: Não Churn | Previsto: Churn |
| --------------- | ------------------: | --------------: |
| Real: Não Churn |                 916 |             119 |
| Real: Churn     |                 160 |             214 |

**Interpretação:**

O modelo identificou corretamente 214 clientes que cancelaram seus serviços.

Entretanto, deixou de identificar 160 clientes que também apresentaram churn.

O Recall de 57,22% indica que aproximadamente 57 em cada 100 clientes que efetivamente cancelaram foram identificados pelo modelo, considerando o limiar de classificação utilizado.

O resultado evidencia oportunidades de melhoria, principalmente para aplicações em que deixar de identificar um cliente com risco de cancelamento represente um custo elevado.

## 8. Disponibilização do modelo

O modelo campeão foi serializado utilizando Joblib e armazenado em:

`models/churn_model.joblib`

A solução utiliza FastAPI para disponibilizar as previsões por meio de uma API REST.

### Endpoints

| Método | Endpoint   | Finalidade                   |
| ------ | ---------- | ---------------------------- |
| GET    | `/`        | Informações da aplicação     |
| GET    | `/health`  | Verificação de funcionamento |
| POST   | `/predict` | Previsão de churn            |

### Exemplo de resposta

```json
{
  "previsao_churn": 1,
  "probabilidade_churn": 0.7893,
  "classificacao": "Churn"
}
```

Nesse exemplo, o modelo estimou uma probabilidade de churn de 78,93% para um cliente fictício.

O valor representa uma estimativa do modelo, não uma certeza de cancelamento.

## 9. Testes automatizados

A API foi submetida a quatro testes automatizados utilizando Pytest e o cliente de testes do FastAPI.

| Teste                                 | Resultado |
| ------------------------------------- | --------- |
| Health Check                          | Aprovado  |
| Previsão com dados válidos            | Aprovado  |
| Rejeição de dados incompletos         | Aprovado  |
| Rejeição de dados numéricos inválidos | Aprovado  |

**Resultado da execução:** 4 testes aprovados.

Os testes verificam comportamentos específicos da aplicação, incluindo disponibilidade, execução de previsões e validação de entradas.

## 10. Limitações conhecidas

O modelo apresenta algumas limitações relevantes.

**Desbalanceamento das classes:** aproximadamente 26,54% dos registros representam clientes que cancelaram seus serviços.

**Falsos negativos:** o modelo deixou de identificar 160 clientes com churn no conjunto de teste.

**Generalização:** o desempenho foi avaliado no dataset utilizado no projeto. Não existem evidências suficientes para afirmar que os resultados seriam reproduzidos em outras empresas ou períodos.

**Dependência das variáveis:** o modelo exige as 20 características utilizadas durante o treinamento.

**Qualidade das probabilidades:** embora o modelo produza probabilidades, não foi realizada uma avaliação específica de calibração probabilística.

**Temporalidade dos dados:** não foi demonstrado que todas as características estariam disponíveis antes do cancelamento em um cenário operacional real.

## 11. Uso recomendado

O modelo pode ser utilizado como ferramenta de apoio à análise de risco de churn e à priorização de ações de retenção.

As previsões devem ser interpretadas como estimativas estatísticas e utilizadas em conjunto com critérios de negócio.

Não é recomendado utilizar o modelo como única fonte para decisões comerciais de alto impacto sem validações adicionais.

## 12. Melhorias futuras

Entre as possibilidades de evolução estão:

* Otimização de hiperparâmetros.
* Ajuste do limiar de classificação para melhorar o Recall.
* Avaliação de técnicas para tratamento do desbalanceamento.
* Análise de importância das variáveis.
* Explicabilidade das previsões.
* Avaliação de calibração das probabilidades.
* Monitoramento de desempenho e mudanças na distribuição dos dados.
* Validação com dados mais recentes e representativos de um ambiente real.
* Implantação da API em ambiente de nuvem.

## 13. Considerações finais

O projeto demonstrou a construção de uma solução de Machine Learning para classificação de churn, contemplando análise exploratória, preparação dos dados, treinamento, comparação de algoritmos, seleção do modelo campeão, serialização e disponibilização por API REST.

A Logistic Regression apresentou o melhor F1-Score médio na validação cruzada entre os modelos avaliados.

A API foi validada por meio de uma requisição de previsão bem-sucedida e quatro testes automatizados aprovados.

Os resultados demonstram a viabilidade técnica da solução como protótipo, embora sua utilização em produção dependa de validações adicionais.
