from pydantic import BaseModel


class QuestionAnswer(BaseModel):
    question: str
    answer: str