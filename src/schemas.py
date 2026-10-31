from pydantic import BaseModel, Field, ConfigDict


class ClienteChurn(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid"
    )

    gender: str = Field(alias="Gender")
    senior_citizen: str = Field(alias="Senior Citizen")
    partner: str = Field(alias="Partner")
    dependents: str = Field(alias="Dependents")

    tenure_months: int = Field(alias="Tenure Months", ge=0, le=72)

    phone_service: str = Field(alias="Phone Service")
    multiple_lines: str = Field(alias="Multiple Lines")
    internet_service: str = Field(alias="Internet Service")
    online_security: str = Field(alias="Online Security")
    online_backup: str = Field(alias="Online Backup")
    device_protection: str = Field(alias="Device Protection")
    tech_support: str = Field(alias="Tech Support")
    streaming_tv: str = Field(alias="Streaming TV")
    streaming_movies: str = Field(alias="Streaming Movies")
    contract: str = Field(alias="Contract")
    paperless_billing: str = Field(alias="Paperless Billing")
    payment_method: str = Field(alias="Payment Method")

    monthly_charges: float = Field(alias="Monthly Charges", ge=0)
    total_charges: float = Field(alias="Total Charges", ge=0)
    cltv: float = Field(alias="CLTV", ge=0)