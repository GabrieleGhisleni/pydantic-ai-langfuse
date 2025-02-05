from __future__ import annotations as _annotations

from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic_ai.models import check_allow_model_requests
from pydantic_ai.models.openai import (
    NOT_GIVEN,
    AsyncStream,
    ChatCompletionChunk,
    OpenAIAgentModel,
    OpenAIModel,
    OpenAIModelSettings,
    chat,
)
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from langfuse.openai import AsyncOpenAI as LanguageFuseAsyncOpenAI
    from openai.types.chat import ChatCompletion
    from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
    from pydantic_ai.tools import ToolDefinition


class LangfuseOpenAIModelSettings(ModelSettings):
    """Additional langfuse settings to use when invoking OpenAI model request.
    listed here: https://langfuse.com/docs/integrations/openai/python/get-started#advanced-usage
    """

    name: str
    "Set name to identify a specific type of generation."

    metadata: dict[str, Any]
    "Set metadata with additional information that you want to see in Langfuse."

    session_id: str
    "The current session."

    user_id: str
    "The current user_id"

    tags: list[str]
    "Set tags to categorize and filter traces."


@dataclass(init=False)
class LangfuseOpenAIModel(OpenAIModel):
    client: LanguageFuseAsyncOpenAI

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> LangfuseOpenAIAgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return LangfuseOpenAIAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
            self.system_prompt_role,
        )


@dataclass
class LangfuseOpenAIAgentModel(OpenAIAgentModel):
    client: LanguageFuseAsyncOpenAI

    @overload
    async def _completions_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[True],
        model_settings: OpenAIModelSettings | LangfuseOpenAIModelSettings,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[False],
        model_settings: OpenAIModelSettings | LangfuseOpenAIModelSettings,
    ) -> ChatCompletion: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIModelSettings | LangfuseOpenAIModelSettings,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        if not self.tools:
            tool_choice: Literal["none", "required", "auto"] | None = None
        elif not self.allow_text_result:
            tool_choice = "required"
        else:
            tool_choice = "auto"

        openai_messages = list(chain(*(self._map_message(m) for m in messages)))
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            n=1,
            parallel_tool_calls=model_settings.get("parallel_tool_calls", NOT_GIVEN),
            tools=self.tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
            stream_options={"include_usage": True} if stream else NOT_GIVEN,
            max_tokens=model_settings.get("max_tokens", NOT_GIVEN),
            temperature=model_settings.get("temperature", NOT_GIVEN),
            top_p=model_settings.get("top_p", NOT_GIVEN),
            timeout=model_settings.get("timeout", NOT_GIVEN),
            seed=model_settings.get("seed", NOT_GIVEN),
            presence_penalty=model_settings.get("presence_penalty", NOT_GIVEN),
            frequency_penalty=model_settings.get("frequency_penalty", NOT_GIVEN),
            logit_bias=model_settings.get("logit_bias", NOT_GIVEN),
            name=model_settings.get("name"),
            metadata=model_settings.get("metadata"),
            session_id=model_settings.get("session_id"),
            user_id=model_settings.get("user_id"),
            tags=model_settings.get("tags"),
        )
