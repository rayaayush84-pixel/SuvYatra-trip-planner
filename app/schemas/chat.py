from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class TripContext(BaseModel):
    """The user's active SuvYatra trip, when there is one.

    All fields are optional so older clients that send no trip context
    keep working unchanged.
    """

    destination: str | None = Field(default=None, max_length=200)
    start_date: str | None = Field(default=None, max_length=50)
    end_date: str | None = Field(default=None, max_length=50)
    dates: str | None = Field(default=None, max_length=200)
    travellers: str | None = Field(default=None, max_length=200)
    budget: str | None = Field(default=None, max_length=200)
    preferences: str | None = Field(default=None, max_length=1000)
    hotels: list[str] | None = None
    activities: list[str] | None = None
    transportation: str | None = Field(default=None, max_length=500)
    itinerary: str | None = Field(default=None, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Message] = Field(default_factory=list, max_length=20)
    trip_context: TripContext | None = None


class Source(BaseModel):
    source: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
