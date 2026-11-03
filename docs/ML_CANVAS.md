# ML Canvas — Predição de Churn de Clientes

**Projeto:** Tech Challenge — Fase 1
**Instituição:** FIAP — Pós-Tech Machine Learning Engineering
**Tema:** Pipeline Preditivo de Churn com Modelagem Clássica e API REST
**Tipo de problema:** Classificação binária supervisionada

---

## 1. Problema de negócio

Uma operadora de telecomunicações enfrenta o desafio de identificar clientes que apresentam risco de cancelar seus serviços.

O cancelamento de clientes, conhecido como *churn*, pode provocar perda de receita recorrente, aumento dos custos de aquisição de novos consumidores e redução da base ativa de clientes.

Sem uma abordagem preditiva, a identificação de clientes em risco pode ocorrer somente após a manifestação de insatisfação ou o próprio cancelamento.

**Problema central:** como identificar antecipadamente clientes com maior probabilidade de churn, utilizando informações contratuais, financeiras e relacionadas aos serviços contratados?

## 2. Objetivo da solução

Desenvolver uma solução de Machine Learning capaz de estimar a probabilidade de cancelamento de clientes de uma operadora de telecomunicações.

A solução deverá:

* Receber informações cadastrais, contratuais e financeiras de um cliente.
* Aplicar o mesmo pré-processamento utilizado durante o treinamento.
* Estimar a probabilidade de churn.
* Retornar uma classificação indicando risco de cancelamento.
* Disponibilizar as previsões por meio de uma API REST.

**Resultado esperado:** fornecer informações preditivas que possam apoiar a priorização de estratégias de retenção.

## 3. Stakeholders

| Stakeholder                        | Interesse na solução                                                         |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| Diretoria                          | Reduzir perdas associadas ao cancelamento e apoiar decisões estratégicas.    |
| Equipe de retenção                 | Identificar clientes com maior risco para priorizar ações de relacionamento. |
| Marketing e CRM                    | Planejar campanhas direcionadas aos segmentos identificados.                 |
| Atendimento ao cliente             | Utilizar informações de risco como apoio à abordagem de relacionamento.      |
| Equipe de dados e Machine Learning | Desenvolver, avaliar, documentar e monitorar o modelo preditivo.             |
| Engenharia de software             | Integrar as previsões aos sistemas corporativos.                             |

Os stakeholders representam os papéis previstos em um cenário empresarial hipotético, e não equipes reais envolvidas no desenvolvimento acadêmico.

## 4. Fonte de dados

**Dataset:** IBM Telco Customer Churn.

**Origem:** conjunto público de dados disponibilizado para estudos de comportamento de clientes de telecomunicações.

**Volume:** 7.043 registros.

**Estrutura original:** 33 colunas.

O conjunto contém informações relacionadas a:

* Perfil dos clientes.
* Tempo de permanência.
* Tipo de contrato.
* Método de pagamento.
* Serviços de telefonia e internet.
* Suporte técnico e segurança online.
* Cobranças mensais e totais.
* Customer Lifetime Value (CLTV).
* Situação de cancelamento.

Durante a preparação, foram identificadas inconsistências na variável `Total Charges` e variáveis que não deveriam ser utilizadas como preditoras por apresentarem risco de vazamento de informações.

## 5. Variável-alvo

**Variável:** `Churn Label`.

**Tipo:** classificação binária.

| Classe | Significado             |
| ------ | ----------------------- |
| `No`   | Cliente que permaneceu. |
| `Yes`  | Cliente que cancelou.   |

A análise exploratória identificou:

* **5.174 clientes sem churn:** 73,46%.
* **1.869 clientes com churn:** 26,54%.

A distribuição demonstra predominância de clientes que permaneceram, exigindo atenção à escolha das métricas de avaliação.

## 6. Abordagem de Machine Learning

A solução utiliza aprendizado supervisionado para classificação binária.

Foram avaliados três algoritmos:

1. **Regressão Logística:** utilizada como modelo baseline.
2. **Random Forest:** modelo baseado em conjuntos de árvores de decisão.
3. **MLPClassifier:** rede neural artificial do tipo Multilayer Perceptron.

O desenvolvimento contempla:

* Análise exploratória.
* Tratamento de inconsistências.
* Transformação de variáveis categóricas.
* Separação entre treinamento e teste.
* Validação cruzada.
* Comparação dos algoritmos.
* Seleção do modelo campeão.
* Persistência do pipeline treinado.

**Modelo selecionado:** Regressão Logística.

A seleção considerou principalmente o desempenho médio de F1-Score na validação cruzada.

## 7. Métricas técnicas

A avaliação utiliza diferentes métricas para compreender o desempenho do modelo.

| Métrica   | Finalidade                                                                       |
| --------- | -------------------------------------------------------------------------------- |
| Accuracy  | Medir a proporção total de classificações corretas.                              |
| Precision | Medir a proporção de previsões positivas que correspondem a cancelamentos reais. |
| Recall    | Medir a capacidade de identificar clientes que efetivamente cancelaram.          |
| F1-Score  | Equilibrar Precision e Recall.                                                   |
| ROC-AUC   | Avaliar a capacidade de discriminação entre as classes.                          |

**Métrica principal de seleção:** F1-Score médio na validação cruzada.

### Resultados do modelo campeão

| Métrica   | Resultado no teste |
| --------- | -----------------: |
| Accuracy  |             0,8020 |
| Precision |             0,6426 |
| Recall    |             0,5722 |
| F1-Score  |             0,6054 |
| ROC-AUC   |             0,8489 |

**F1-Score médio na validação cruzada:** 0,6213.

Esses resultados representam a avaliação realizada no conjunto de dados do projeto, não uma garantia de desempenho futuro em produção.

## 8. Métricas de negócio

O objetivo de negócio é apoiar ações de retenção e reduzir perdas associadas ao cancelamento.

Como a solução foi desenvolvida em contexto acadêmico, não foram realizadas campanhas reais de retenção. Portanto, as métricas abaixo são propostas para uma eventual aplicação empresarial.

### Métrica principal — Taxa de retenção dos clientes abordados

**Definição:** percentual de clientes classificados como alto risco e abordados por uma campanha que permanecem ativos após um período de acompanhamento previamente definido.

**Fórmula:**

Taxa de retenção = (Clientes abordados que permaneceram ativos / Total de clientes abordados) × 100

Essa métrica deve ser acompanhada com um grupo de controle para avaliar se a campanha produziu um efeito adicional de retenção.

### Métricas complementares

**Taxa de churn do público monitorado:** proporção de clientes que cancelaram em determinado período.

**Custo por cliente retido:** custo total da campanha dividido pela quantidade de clientes cuja retenção adicional foi estimada.

**Retorno financeiro incremental:** benefício econômico estimado das ações de retenção, descontados seus custos.

**Indicador operacional:** percentual de clientes classificados como alto risco que foram efetivamente abordados pela equipe responsável.

**Observação:** essas métricas são propostas de acompanhamento e não resultados já alcançados pelo projeto.

## 9. Implantação e utilização

A solução foi desenvolvida utilizando Python e o ecossistema Scikit-Learn.

O modelo selecionado foi persistido no arquivo:

`models/churn_model.joblib`

A inferência é disponibilizada por uma API REST desenvolvida com FastAPI.

### Endpoints

**GET `/health`**

Permite verificar se a API está disponível.

**POST `/predict`**

Recebe os dados de um cliente e retorna:

* Classificação prevista.
* Probabilidade estimada de churn.

A solução foi validada localmente por meio do Swagger e de quatro testes automatizados com Pytest.

O deploy em nuvem não faz parte da implementação atual.

## 10. Riscos, limitações e cuidados

### Qualidade e representatividade dos dados

O modelo foi treinado com um dataset público, que pode não representar o comportamento de clientes de uma operadora real.

### Desbalanceamento das classes

A classe de cancelamento representa aproximadamente 26,54% dos registros. Por esse motivo, a acurácia isoladamente não é suficiente para avaliar o desempenho.

### Falsos negativos

Clientes que efetivamente cancelariam podem ser classificados como clientes sem risco de churn.

Isso pode reduzir a capacidade de direcionar ações preventivas.

### Falsos positivos

Clientes que permaneceriam podem ser classificados como clientes com risco de cancelamento.

Isso pode gerar custos desnecessários em campanhas de retenção.

### Vazamento de informações

Variáveis que revelam diretamente o cancelamento ou contêm informações posteriores ao evento não devem ser utilizadas como preditoras.

### Possíveis vieses

O modelo pode apresentar diferenças de desempenho entre segmentos de clientes. Essas diferenças precisariam ser avaliadas antes de uma aplicação real.

### Monitoramento

Em um ambiente produtivo, seria necessário acompanhar alterações no comportamento dos clientes, na distribuição dos dados e no desempenho preditivo.

### Interpretação das previsões

A probabilidade retornada pelo modelo representa uma estimativa estatística, não uma certeza de cancelamento.

## 11. Critérios de sucesso do projeto

### Sucesso técnico

* Pipeline de pré-processamento e classificação funcional.
* Comparação documentada de três algoritmos.
* Modelo campeão selecionado com critérios explícitos.
* Modelo treinado salvo para reutilização.
* API REST funcional.
* Testes automatizados aprovados.
* Código e documentação disponíveis no GitHub.

### Sucesso de negócio

Em uma aplicação futura, o sucesso dependeria de demonstrar que o uso das previsões contribui para priorizar ações de retenção e melhorar resultados em comparação com uma abordagem sem o modelo.

Essa validação exigiria acompanhamento temporal, definição de custos e avaliação experimental das campanhas.

## 12. Considerações finais

O projeto demonstra a aplicação de um pipeline de Machine Learning para previsão de churn, integrando entendimento do problema de negócio, análise de dados, modelagem preditiva e disponibilização das previsões por API REST.

O ML Canvas organiza as decisões centrais da solução e estabelece uma ligação entre os objetivos empresariais e os critérios técnicos de desenvolvimento.

A implementação atual representa uma prova de conceito acadêmica. Sua adoção em ambiente produtivo dependeria de validações adicionais, integração com sistemas corporativos e monitoramento contínuo.
