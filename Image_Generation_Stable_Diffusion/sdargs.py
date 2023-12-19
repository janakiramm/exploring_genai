import typing as t

from pydantic import BaseModel


class SDArgs(BaseModel):
    prompt: str
    negative_prompt: t.Optional[str] = None
    height: t.Optional[int] = 512
    width: t.Optional[int] = 512

    class Config:
        extra = "allow"