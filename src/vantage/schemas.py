from pydantic import BaseModel, Field
import numpy as np

class CustomerContext(BaseModel):
    segment: str = Field(description="Customer segment: 'student', 'professional', or 'default'")
    day_type: str = Field(description="Day type: 'weekday' or 'weekend'")
    competitor_price: float = Field(description="Competitor's price for the product")

    def to_vector(self) -> np.ndarray:
        """
        Encodes the context into a numeric vector:
        [intercept, is_student, is_professional, is_weekend, competitor_price]
        """
        is_student = 1.0 if self.segment == "student" else 0.0
        is_professional = 1.0 if self.segment == "professional" else 0.0
        is_weekend = 1.0 if self.day_type == "weekend" else 0.0
        return np.array([
            1.0,
            is_student,
            is_professional,
            is_weekend,
            self.competitor_price
        ], dtype=np.float64)

class GroundTruthEntry(BaseModel):
    segment: str
    day_type: str
    competitor_price: float
    optimal_price: float
    expected_revenue: float
