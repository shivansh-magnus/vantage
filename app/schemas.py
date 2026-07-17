from pydantic import BaseModel, Field

class CustomerContextInput(BaseModel):
    segment: str = Field(..., description="Customer segment: 'student', 'professional', or 'default'")
    day_type: str = Field(..., description="Day type: 'weekday' or 'weekend'")
    competitor_price: float = Field(..., description="Continuous competitor price in dollars")

class PriceResponse(BaseModel):
    price: float = Field(..., description="Price suggested by the bandit agent")
    arm_id: int = Field(..., description="Index of the selected price arm")
    request_id: str = Field(..., description="Unique transaction ID to map this offer to its outcome")

class OutcomeInput(BaseModel):
    request_id: str = Field(..., description="Unique transaction ID of the original price offer")
    purchased: bool = Field(..., description="True if customer purchased, False otherwise")

class Acknowledgement(BaseModel):
    status: str = Field(..., description="Status of the request, e.g., 'success' or 'error'")
    message: str = Field(..., description="Detailed description of the transaction result")
