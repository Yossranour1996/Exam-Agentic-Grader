# src/agents/agent_base.py
from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain.agents.middleware import AgentMiddleware
from langchain_core.prompts import ChatPromptTemplate


class AgentBase:
    """Base class for all agents, providing a unified interface for initialization and invocation."""

    def __init__(
        self,
        name: str,
        model: str,
        temperature: float,
        max_tokens: int | None,
        tools: list[BaseTool] | None,
        middleware: list[AgentMiddleware] | None,
        response_format: type[BaseModel],
        prompt_template: ChatPromptTemplate,
    ):
        self.name = name
        self.response_format = response_format
        model_client = init_chat_model(model=model, temperature=temperature, max_tokens=max_tokens)
        agent = create_agent(model=model_client, tools=tools or [], middleware=middleware or (), response_format=response_format)
        self.chain = prompt_template | agent

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return JSON, text, and only messages created by this invocation."""
        output = self.chain.invoke(state)
        result = output.get('structured_response')

        if not isinstance(result, self.response_format):
            message = f"{self.name} did not return a valid structured response."
            if logger := state.get('logger'):
                logger.log(message, level="error")
            raise TypeError(message)

        prior = state.get('messages', [])
        messages = [message for message in output.get('messages', []) if message not in prior]
        return {
            "result_json": result.model_dump(),
            "result_str": self.parse_result(result),
            "messages": messages
        }

    @staticmethod
    def parse_result(result: BaseModel) -> str:
        """Render a model-specific structured result for reports and prompts."""
        raise NotImplementedError
