"""
Output parsers and the schemas they parse into.

StrOutputParser just extracts .content from the model's response as plain text.
PydanticOutputParser forces the model to return data matching a schema, then
parses that into a real Python object -- useful whenever an API response needs
a predictable shape instead of free-form text.
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

str_parser = StrOutputParser()


class ChatAnswer(BaseModel):
    """Structured shape for a chat response, used later by the /chat endpoint."""
    answer: str = Field(description="The answer to the user's question")
    confidence: str = Field(description="One of: high, medium, low -- how confident the answer is")


chat_answer_parser = PydanticOutputParser(pydantic_object=ChatAnswer)