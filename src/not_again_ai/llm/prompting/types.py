from abc import ABC, abstractmethod
from collections.abc import Collection, Set
from typing import Literal

from not_again_ai.llm.chat_completion.types import MessageT


class BaseTokenizer(ABC):
    def __init__(
        self,
        model: str,
        provider: str,
        allowed_special: Literal["all"] | Set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] | None = None,
    ):
        self.model = model
        self.provider = provider
        self.allowed_special = allowed_special
        self.disallowed_special = disallowed_special

        self.init_tokenizer(model, provider, allowed_special, disallowed_special)

    @abstractmethod
    def init_tokenizer(
        self,
        model: str,
        provider: str,
        allowed_special: Literal["all"] | Set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] | None = None,
    ) -> None:
        pass

    @abstractmethod
    def truncate_str(self, text: str, max_len: int) -> str:
        pass

    @abstractmethod
    def num_tokens_in_str(self, text: str) -> int:
        pass

    @abstractmethod
    def num_tokens_in_messages(self, messages: list[MessageT]) -> int:
        pass
