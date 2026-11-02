from pydantic import BaseModel, Field, HttpUrl


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    source: str = Field(default="api", max_length=200)


class IngestUrlRequest(BaseModel):
    url: HttpUrl


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int
