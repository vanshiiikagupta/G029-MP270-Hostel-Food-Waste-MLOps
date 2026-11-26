

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    students_registered: int = Field(gt=0)
    attendance_pct: float = Field(ge=0, le=100)
    menu_rating: float
    temp_c: float
    cooked_kg: float = Field(ge=0, le=1000)
    plate_size_cm: float = Field(ge=10)
    meal: str
    hostel: str