from collections.abc import Collection, Set
from typing import Literal

from loguru import logger
import tiktoken

from not_again_ai.llm.chat_completion.types import MessageT
from not_again_ai.llm.prompting.types import BaseTokenizer


class TokenizerOpenAI(BaseTokenizer):
    def __init__(
        self,
        model: str,
        provider: str = "openai",
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
        provider: str = "openai",
        allowed_special: Literal["all"] | Set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] | None = None,
    ) -> None:
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            default_encoding = "o200k_base"
            logger.warning(f"Model {model} not found. Using {default_encoding} encoding.")
            self.encoding = tiktoken.get_encoding(default_encoding)

        # Set defaults if not provided
        if not allowed_special:
            self.allowed_special = set()
        if not disallowed_special:
            self.disallowed_special = ()

    def truncate_str(self, text: str, max_len: int) -> str:
        tokens = self.encoding.encode(
            text,
            allowed_special=self.allowed_special if self.allowed_special is not None else set(),
            disallowed_special=self.disallowed_special if self.disallowed_special is not None else (),
        )
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            truncated_text = self.encoding.decode(tokens)
            return truncated_text
        else:
            return text

    def num_tokens_in_str(self, text: str) -> int:
        return len(
            self.encoding.encode(
                text,
                allowed_special=self.allowed_special if self.allowed_special is not None else set(),
                disallowed_special=self.disallowed_special if self.disallowed_special is not None else (),
            )
        )

    def num_tokens_in_messages(self, messages: list[MessageT]) -> int:
        if self.model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "o1",
            "o1-2024-12-17",
            "o1-mini",
            "o1-mini-2024-09-12",
            "o1-preview",
            "o1-preview-2024-09-12",
        }:
            tokens_per_message = 3  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = 1  # if there's a name, the role is omitted
        elif self.model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1
        else:
            logger.warning(f"Model {self.model} not supported. Assuming gpt-4o encoding.")
            tokens_per_message = 3
            tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            message_dict = message.model_dump(exclude_none=True)
            for key, value in message_dict.items():
                if isinstance(value, str):
                    num_tokens += len(
                        self.encoding.encode(
                            value,
                            allowed_special=self.allowed_special if self.allowed_special is not None else set(),
                            disallowed_special=self.disallowed_special if self.disallowed_special is not None else (),
                        )
                    )
                    if key == "name":
                        num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
