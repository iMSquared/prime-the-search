from __future__ import annotations
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Any, Tuple


class ActionArgs(BaseModel):
    text: str
    span_start: Optional[int]
    span_end: Optional[int]



class Token(BaseModel):
    r"""
    A token in a sequence, with optional metadata.
    text: str
    log_prob: float
    index: int
    bytes: int
    prob: Optional[float] = None
    """
    text: str
    log_prob: float
    index: int
    bytes: int
    prob: Optional[float] = None


class Sequence(BaseModel):
    r"""
    A sequence of tokens, with optional metadata.
    tokens: Dict[int, Token]
    text: str
    char_to_token_map: Dict[int, int] = {}

    info: Optional[Dict[str, Any]] = None
    log_prob: Optional[Dict[str, float]] = None
    action_args: Optional[List[ActionArgs]] = None
    value: Optional[float] = None
    """
    text: str
    tokens: Optional[Dict[int, Token]] = {}
    char_to_token_map: Optional[Dict[int, int]] = {}

    info: Optional[Dict[str, Any]] = None
    log_prob: Optional[Dict[str, float]] = None
    action_args: Optional[List[ActionArgs]] = None
    value: Optional[float] = None


