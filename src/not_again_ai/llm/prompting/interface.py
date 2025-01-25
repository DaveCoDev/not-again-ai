from collections.abc import Collection, Set
from typing import Literal

from loguru import logger

from not_again_ai.llm.chat_completion.types import MessageT
from not_again_ai.llm.prompting.providers.openai_tiktoken import TokenizerOpenAI
from not_again_ai.llm.prompting.types import BaseTokenizer


class Tokenizer(BaseTokenizer):
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

    def init_tokenizer(
        self,
        model: str,
        provider: str,
        allowed_special: Literal["all"] | Set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] | None = None,
    ) -> None:
        if provider == "openai" or provider == "azure_openai":
            self.tokenizer = TokenizerOpenAI(model, provider, allowed_special, disallowed_special)
        else:
            logger.warning(f"Provider {provider} not supported. Initializing using tiktoken and gpt-4o.")
            self.tokenizer = TokenizerOpenAI("gpt-4o", "openai", allowed_special, disallowed_special)

    def truncate_str(self, text: str, max_len: int) -> str:
        return self.tokenizer.truncate_str(text, max_len)

    def num_tokens_in_str(self, text: str) -> int:
        return self.tokenizer.num_tokens_in_str(text)

    def num_tokens_in_messages(self, messages: list[MessageT]) -> int:
        return self.tokenizer.num_tokens_in_messages(messages)
