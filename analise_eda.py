from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_DATASET = PASTA_PROJETO / "Telco_customer_churn.xlsx"


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

df = pd.read_excel(CAMINHO_DATASET)

print("\n============================================================")
print("ANÁLISE EXPLORATÓRIA DOS DADOS - CHURN")
print("============================================================")

print(f"\nDataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")


# ============================================================
# 3. VISÃO GERAL DO DATASET
# ============================================================

print("\n--- VISÃO GERAL ---")

print(f"Quantidade de clientes: {df.shape[0]}")
print(f"Quantidade de colunas: {df.shape[1]}")
print(f"Linhas duplicadas: {df.duplicated().sum()}")

print("\nPrimeiros registros:")
print(df.head())


# ============================================================
# 4. DISTRIBUIÇÃO DO TARGET
# ============================================================

print("\n--- DISTRIBUIÇÃO DO CHURN ---")

churn_counts = df["Churn Label"].value_counts()

churn_percentual = (
    df["Churn Label"]
    .value_counts(normalize=True)
    .mul(100)
)

distribuicao_churn = pd.DataFrame({
    "Quantidade": churn_counts,
    "Percentual": churn_percentual.round(2)
})

print(distribuicao_churn)

churn_counts.plot(
    kind="bar"
)

plt.title("Distribuição de Churn")
plt.xlabel("Churn")
plt.ylabel("Quantidade de clientes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 5. CHURN POR TIPO DE CONTRATO
# ============================================================

print("\n--- CHURN POR TIPO DE CONTRATO ---")

contract_churn = (
    pd.crosstab(
        df["Contract"],
        df["Churn Label"],
        normalize="index"
    )
    .mul(100)
)

print(contract_churn.round(2))

contract_churn["Yes"].sort_values().plot(
    kind="barh"
)

plt.title("Taxa de Churn por Tipo de Contrato")
plt.xlabel("Taxa de Churn (%)")
plt.ylabel("Tipo de contrato")
plt.tight_layout()
plt.show()


# ============================================================
# 6. CHURN POR MÉTODO DE PAGAMENTO
# ============================================================

print("\n--- CHURN POR MÉTODO DE PAGAMENTO ---")

payment_churn = (
    pd.crosstab(
        df["Payment Method"],
        df["Churn Label"],
        normalize="index"
    )
    .mul(100)
)

print(payment_churn.round(2))

payment_churn["Yes"].sort_values().plot(
    kind="barh"
)

plt.title("Taxa de Churn por Método de Pagamento")
plt.xlabel("Taxa de Churn (%)")
plt.ylabel("Método de pagamento")
plt.tight_layout()
plt.show()


# ============================================================
# 7. CHURN POR SERVIÇO DE INTERNET
# ============================================================

print("\n--- CHURN POR SERVIÇO DE INTERNET ---")

internet_churn = (
    pd.crosstab(
        df["Internet Service"],
        df["Churn Label"],
        normalize="index"
    )
    .mul(100)
)

print(internet_churn.round(2))

internet_churn["Yes"].sort_values().plot(
    kind="barh"
)

plt.title("Taxa de Churn por Serviço de Internet")
plt.xlabel("Taxa de Churn (%)")
plt.ylabel("Serviço de internet")
plt.tight_layout()
plt.show()


# ============================================================
# 8. CHURN POR SUPORTE TÉCNICO
# ============================================================

print("\n--- CHURN POR SUPORTE TÉCNICO ---")

tech_support_churn = (
    pd.crosstab(
        df["Tech Support"],
        df["Churn Label"],
        normalize="index"
    )
    .mul(100)
)

print(tech_support_churn.round(2))

tech_support_churn["Yes"].sort_values().plot(
    kind="barh"
)

plt.title("Taxa de Churn por Suporte Técnico")
plt.xlabel("Taxa de Churn (%)")
plt.ylabel("Suporte técnico")
plt.tight_layout()
plt.show()


# ============================================================
# 9. CHURN POR SEGURANÇA ONLINE
# ============================================================

print("\n--- CHURN POR SEGURANÇA ONLINE ---")

security_churn = (
    pd.crosstab(
        df["Online Security"],
        df["Churn Label"],
        normalize="index"
    )
    .mul(100)
)

print(security_churn.round(2))

security_churn["Yes"].sort_values().plot(
    kind="barh"
)

plt.title("Taxa de Churn por Segurança Online")
plt.xlabel("Taxa de Churn (%)")
plt.ylabel("Segurança online")
plt.tight_layout()
plt.show()


# ============================================================
# 10. TEMPO DE PERMANÊNCIA X CHURN
# ============================================================

print("\n--- TEMPO DE PERMANÊNCIA X CHURN ---")

tenure_stats = (
    df.groupby("Churn Label")["Tenure Months"]
    .describe()
    .round(2)
)

print(tenure_stats)

df.boxplot(
    column="Tenure Months",
    by="Churn Label"
)

plt.title("Tempo de Permanência por Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Meses de permanência")
plt.tight_layout()
plt.show()


# ============================================================
# 11. COBRANÇA MENSAL X CHURN
# ============================================================

print("\n--- COBRANÇA MENSAL X CHURN ---")

monthly_stats = (
    df.groupby("Churn Label")["Monthly Charges"]
    .describe()
    .round(2)
)

print(monthly_stats)

df.boxplot(
    column="Monthly Charges",
    by="Churn Label"
)

plt.title("Cobrança Mensal por Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Cobrança mensal")
plt.tight_layout()
plt.show()


# ============================================================
# 12. CLTV X CHURN
# ============================================================

print("\n--- CLTV X CHURN ---")

cltv_stats = (
    df.groupby("Churn Label")["CLTV"]
    .describe()
    .round(2)
)

print(cltv_stats)

df.boxplot(
    column="CLTV",
    by="Churn Label"
)

plt.title("CLTV por Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("CLTV")
plt.tight_layout()
plt.show()


# ============================================================
# 13. RESUMO FINAL
# ============================================================

print("\n============================================================")
print("EDA CONCLUÍDA")
print("============================================================")

print("\nAnálises realizadas:")
print("- Distribuição do target")
print("- Tipo de contrato x churn")
print("- Método de pagamento x churn")
print("- Serviço de internet x churn")
print("- Suporte técnico x churn")
print("- Segurança online x churn")
print("- Tempo de permanência x churn")
print("- Cobrança mensal x churn")
print("- CLTV x churn")

print("\nAnalise as tabelas e gráficos antes de iniciar a modelagem.")