# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from collections.abc import AsyncIterable
from typing import Any

import aiohttp
from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    BaseAgent,
    Content,
    Message,
    normalize_messages,
)
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

"""
Azure AI Foundry - HuggingFace Text Generation Model Example

This sample demonstrates how to create an agent that uses a HuggingFace
text-generation model (such as microsoft/BioGPT-Large) deployed in Azure AI
Foundry as a Managed Online Endpoint.

Text-generation models differ from chat completion models:
- They take a raw text prompt and return a text continuation.
- They do not maintain a conversation history on the service side.
- They typically do not support tool/function calling.

BioGPT-Large is a domain-specific language model pre-trained on biomedical
literature. It is well-suited for tasks such as:
- Biomedical text generation and sentence completion
- Biomedical question answering (generative style)
- Biomedical relation extraction when prompted correctly

When deployed as an Azure AI Foundry Managed Online Endpoint, the model
exposes a REST endpoint that accepts the standard HuggingFace Inference API
request format and returns generated text.

Environment variables required:
  AZURE_ML_ENDPOINT_URL  - The scoring URL for your deployed endpoint,
                           e.g. https://<name>.<region>.inference.ml.azure.com/score
  AZURE_ML_API_KEY       - (Optional) The primary or secondary key for
                           key-based authentication. When not set, Azure CLI
                           credential (az login) is used instead.

Setup:
  1. Open Azure AI Foundry (https://ai.azure.com) and navigate to your project.
  2. Go to "Models + endpoints" and deploy the microsoft/BioGPT-Large model
     from the HuggingFace model catalog using Managed Compute.
  3. Once deployed, copy the endpoint's REST URL (the "Score" URL) and, if
     using key-based auth, copy one of the endpoint keys.
  4. Set the environment variables above (or add them to your .env file).
"""

# Scope used when obtaining an Azure AD token for Azure ML endpoints.
_AZURE_ML_TOKEN_SCOPE = "https://ml.azure.com/.default"


class FoundryTextGenerationAgent(BaseAgent):
    """An agent that wraps a HuggingFace text-generation model deployed in Azure AI Foundry.

    This agent calls the Azure ML Online Endpoint REST API directly. The endpoint
    must be deployed using the HuggingFace "text-generation" task, which means
    the scoring endpoint accepts requests in the format::

        {"inputs": "<prompt>", "parameters": {...}}

    and returns responses in the format::

        [{"generated_text": "<prompt><continuation>"}]

    Authentication can be either key-based (``api_key``) or credential-based
    (``credential``). When both are provided, the API key takes precedence.

    Examples:
        Using key-based authentication::

            agent = FoundryTextGenerationAgent(
                endpoint_url="https://my-endpoint.region.inference.ml.azure.com/score",
                api_key="my-endpoint-key",
                name="BioGPTAgent",
                instructions="You are a biomedical text generation assistant.",
            )
            result = await agent.run("The mechanism of action of aspirin is")

        Using Azure credential::

            async with AzureCliCredential() as credential:
                agent = FoundryTextGenerationAgent(
                    endpoint_url="https://my-endpoint.region.inference.ml.azure.com/score",
                    credential=credential,
                    name="BioGPTAgent",
                )
                result = await agent.run("SARS-CoV-2 infects human cells by")
    """

    def __init__(
        self,
        endpoint_url: str,
        *,
        credential: AsyncTokenCredential | None = None,
        api_key: str | None = None,
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize a FoundryTextGenerationAgent.

        Args:
            endpoint_url: The REST scoring URL of the deployed Azure ML Online Endpoint,
                e.g. ``https://<endpoint-name>.<region>.inference.ml.azure.com/score``.
            credential: An async Azure credential (e.g. ``AzureCliCredential``).
                Used when ``api_key`` is not provided.
            api_key: The endpoint key for key-based authentication. When set, this
                takes precedence over ``credential``.
            max_new_tokens: Maximum number of new tokens for the model to generate.
                Defaults to 128.
            **kwargs: Additional arguments forwarded to ``BaseAgent``, e.g. ``name``,
                ``description``, ``instructions``.
        """
        super().__init__(**kwargs)
        self._endpoint_url = endpoint_url
        self._credential = credential
        self._api_key = api_key
        self._max_new_tokens = max_new_tokens

    async def _get_auth_headers(self) -> dict[str, str]:
        """Build the Authorization header for the endpoint request.

        Supports both key-based and Azure AD token-based authentication.

        Returns:
            A dict containing the ``Authorization`` and ``Content-Type`` headers.

        Raises:
            ValueError: When neither ``api_key`` nor ``credential`` is provided.
        """
        if self._api_key:
            # Key-based auth: use the endpoint key as a Bearer token.
            return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if self._credential:
            # Token-based auth: obtain an Azure AD token for the ML scope.
            token = await self._credential.get_token(_AZURE_ML_TOKEN_SCOPE)
            return {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}
        raise ValueError(
            "Authentication is required. Provide either 'api_key' or 'credential' when "
            "creating FoundryTextGenerationAgent."
        )

    def run(
        self,
        messages: str | Message | list[str] | list[Message] | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> "AsyncIterable[AgentResponseUpdate] | asyncio.Future[AgentResponse]":
        """Run the agent and return a response.

        For text-generation models, only non-streaming responses are supported
        natively. When ``stream=True`` is requested, the agent simulates streaming
        by yielding the response word-by-word after obtaining the full completion.

        Args:
            messages: The input prompt. For text-generation, the last user message
                (or the raw string) is used directly as the text prompt.
            stream: If ``True``, return an async iterable of word-by-word updates.
                If ``False`` (default), return an awaitable ``AgentResponse``.
            session: Conversation session (optional). When provided, full message
                history is stored in ``session.state`` for reference.
            **kwargs: Additional keyword arguments (forwarded to the underlying call).

        Returns:
            An awaitable ``AgentResponse`` when ``stream=False``, or an async
            iterable of ``AgentResponseUpdate`` instances when ``stream=True``.
        """
        if stream:
            return self._run_stream(messages=messages, session=session, **kwargs)
        return self._run(messages=messages, session=session, **kwargs)

    def _build_prompt(self, messages: list[Message]) -> str:
        """Extract the text prompt from a list of messages.

        For text-generation models there is no chat history concept. This helper
        uses the agent's ``instructions`` as a prefix (when available) and the
        last user message as the actual prompt to complete.

        Args:
            messages: Normalized list of ``Message`` objects.

        Returns:
            A single string prompt to send to the text-generation endpoint.
        """
        # Find the last user message to use as the prompt.
        user_text = ""
        for msg in reversed(messages):
            if msg.role == "user" and msg.text:
                user_text = msg.text
                break

        # Optionally prepend agent instructions to guide the model.
        if self.instructions and user_text:
            return f"{self.instructions}\n\n{user_text}"
        return user_text or (messages[-1].text if messages else "")

    async def _call_endpoint(self, prompt: str) -> str:
        """Call the Azure ML Online Endpoint and return the generated text.

        Args:
            prompt: The text prompt to send to the model.

        Returns:
            The generated text (which may include the original prompt depending
            on the model's configuration).

        Raises:
            aiohttp.ClientResponseError: When the endpoint returns a non-2xx status.
        """
        headers = await self._get_auth_headers()

        # Standard HuggingFace text-generation inference API payload.
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self._max_new_tokens,
            },
        }

        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(self._endpoint_url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                result: list[dict[str, Any]] = await resp.json()

        # HuggingFace text-generation response format:
        # [{"generated_text": "<original prompt><generated continuation>"}]
        return result[0]["generated_text"]

    async def _run(
        self,
        messages: str | Message | list[str] | list[Message] | None = None,
        *,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """Non-streaming implementation: call the endpoint and return the full response."""
        normalized = normalize_messages(messages)
        prompt = self._build_prompt(normalized)

        generated_text = await self._call_endpoint(prompt)

        response_message = Message(role="assistant", contents=[Content.from_text(generated_text)])

        # Optionally store the conversation in session state.
        if session is not None:
            stored: list[Message] = (
                session.state.setdefault("text_generation_messages", [])
            )
            stored.extend(normalized)
            stored.append(response_message)

        return AgentResponse(messages=[response_message])

    async def _run_stream(
        self,
        messages: str | Message | list[str] | list[Message] | None = None,
        *,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentResponseUpdate]:
        """Streaming implementation: yield the generated text word-by-word.

        Note:
            The BioGPT-Large managed endpoint does not natively stream tokens.
            This implementation obtains the full completion first, then yields
            it word-by-word to simulate a streaming experience compatible with
            the Agent Framework streaming API.
        """
        normalized = normalize_messages(messages)
        prompt = self._build_prompt(normalized)

        generated_text = await self._call_endpoint(prompt)

        # Simulate streaming by yielding the response word-by-word.
        words = generated_text.split()
        for i, word in enumerate(words):
            chunk = f" {word}" if i > 0 else word
            yield AgentResponseUpdate(
                contents=[Content.from_text(chunk)],
                role="assistant",
            )

        # Optionally store the conversation in session state.
        if session is not None:
            response_message = Message(role="assistant", contents=[Content.from_text(generated_text)])
            stored: list[Message] = (
                session.state.setdefault("text_generation_messages", [])
            )
            stored.extend(normalized)
            stored.append(response_message)


async def main() -> None:
    """Demonstrate using BioGPT-Large deployed in Azure AI Foundry as an agent.

    This example uses Azure CLI credential (az login). To use an endpoint key
    instead, set AZURE_ML_API_KEY in your environment and pass it as ``api_key``.
    """
    print("=== Azure AI Foundry - HuggingFace Text Generation Agent Example ===\n")

    # Read configuration from environment variables.
    endpoint_url = os.environ["AZURE_ML_ENDPOINT_URL"]
    api_key = os.environ.get("AZURE_ML_API_KEY")  # Optional; falls back to Azure CLI credential.

    # --- Example 1: Non-streaming response ---
    print("--- Non-streaming Example ---")

    if api_key:
        # Key-based auth: no credential context manager needed.
        agent = FoundryTextGenerationAgent(
            endpoint_url=endpoint_url,
            api_key=api_key,
            name="BioGPTAgent",
            description="Biomedical text generation agent powered by BioGPT-Large.",
            instructions="Complete the following biomedical text:",
            max_new_tokens=100,
        )
        prompt = "Aspirin inhibits the synthesis of prostaglandins by"
        print(f"Prompt: {prompt}")
        result = await agent.run(prompt)
        print(f"Generated: {result.messages[0].text}\n")
    else:
        # Token-based auth: use AzureCliCredential (requires `az login`).
        async with AzureCliCredential() as credential:
            agent = FoundryTextGenerationAgent(
                endpoint_url=endpoint_url,
                credential=credential,
                name="BioGPTAgent",
                description="Biomedical text generation agent powered by BioGPT-Large.",
                instructions="Complete the following biomedical text:",
                max_new_tokens=100,
            )
            prompt = "Aspirin inhibits the synthesis of prostaglandins by"
            print(f"Prompt: {prompt}")
            result = await agent.run(prompt)
            print(f"Generated: {result.messages[0].text}\n")

    # --- Example 2: Streaming response ---
    print("--- Streaming Example ---")

    if api_key:
        agent = FoundryTextGenerationAgent(
            endpoint_url=endpoint_url,
            api_key=api_key,
            name="BioGPTAgent",
            max_new_tokens=80,
        )
        prompt = "The BRCA1 gene plays a critical role in"
        print(f"Prompt: {prompt}")
        print("Generated: ", end="", flush=True)
        async for chunk in agent.run(prompt, stream=True):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print("\n")
    else:
        async with AzureCliCredential() as credential:
            agent = FoundryTextGenerationAgent(
                endpoint_url=endpoint_url,
                credential=credential,
                name="BioGPTAgent",
                max_new_tokens=80,
            )
            prompt = "The BRCA1 gene plays a critical role in"
            print(f"Prompt: {prompt}")
            print("Generated: ", end="", flush=True)
            async for chunk in agent.run(prompt, stream=True):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print("\n")


if __name__ == "__main__":
    asyncio.run(main())


"""
Sample output (illustrative - actual output depends on the deployed model):

=== Azure AI Foundry - HuggingFace Text Generation Agent Example ===

--- Non-streaming Example ---
Prompt: Aspirin inhibits the synthesis of prostaglandins by
Generated: Aspirin inhibits the synthesis of prostaglandins by irreversibly inhibiting
cyclooxygenase (COX) enzymes, specifically COX-1 and COX-2. This prevents the
conversion of arachidonic acid to prostaglandin H2, a precursor to various
pro-inflammatory prostaglandins and thromboxane A2.

--- Streaming Example ---
Prompt: The BRCA1 gene plays a critical role in
Generated: The BRCA1 gene plays a critical role in DNA damage repair, particularly in
homologous recombination. Mutations in BRCA1 significantly increase the risk of
breast and ovarian cancers.
"""
