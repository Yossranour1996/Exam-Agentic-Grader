# other_architectures/single_agent/agents/agent_base.py
"""Shared base class for agents that use LangChain prompts and structured responses."""

from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain.agents.middleware import AgentMiddleware
from langchain_core.prompts import ChatPromptTemplate


class AgentBase:
    """Base class for agents that wrap a prompt template with structured output."""

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
        """Execute the prompt chain and return structured output plus new messages."""
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
        """Render a structured model result into human-readable text for logs or reports."""
        raise NotImplementedError
