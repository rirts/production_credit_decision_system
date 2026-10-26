from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

class CreditInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal['A11', 'A12', 'A13', 'A14']
    duration: int = Field(gt=0, le=120)
    credit_history: Literal['A30','A31','A32','A33','A34']
    purpose: Literal['A40','A41','A42','A43','A44','A45','A46','A48','A49','A410']
    credit_amount: int = Field(gt=0, le=1000000)
    savings: Literal['A61','A62','A63','A64','A65']
    employment_duration: Literal['A71','A72','A73','A74','A75']
    installment_rate: int = Field(ge=1, le=4)
    personal_status_sex: Literal['A91','A92','A93','A94','A95']
    other_debtors: Literal['A101','A102','A103']
    present_residence: int = Field(ge=1, le=4)
    property: Literal['A121','A122','A123','A124']
    age: int = Field(ge=18, le=120)
    other_installment_plans: Literal['A141','A142','A143']
    housing: Literal['A151','A152','A153']
    number_of_credits: int = Field(ge=1, le=4)
    job: Literal['A171','A172','A173','A174']
    people_liable: int = Field(ge=1, le=2)
    telephone: Literal['A191','A192']
    foreign_worker: Literal['A201','A202']
