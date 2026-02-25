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
    the scoring endpoint accepts requests in the HuggingFace Inference API format::

        {"inputs": "<prompt>", "parameters": {...}}

    and returns responses in the format::

        [{"generated_text": "<prompt><continuation>"}]

    The generation parameters mirror the HuggingFace ``transformers`` library's
    ``model.generate()`` API (see https://github.com/microsoft/BioGPT). Key
    parameters include:

    - ``max_new_tokens`` / ``min_new_tokens``: Control the number of tokens to generate.
    - ``temperature``, ``top_k``, ``top_p``: Sampling parameters.
    - ``do_sample``: Whether to use sampling instead of greedy decoding.
    - ``repetition_penalty``: Penalty for repeating tokens.

    **Advanced parameters (require larger compute):**

    - ``num_beams``: Number of beams for beam search (requires more memory).
    - ``early_stopping``: Stop beam search early when criteria are met.

    Note: Beam search parameters (``num_beams``, ``early_stopping``) require
    larger compute instances (e.g., Standard_DS3_v2 or GPU SKUs). Small compute
    instances may return 424 errors when these parameters are used.

    Authentication can be either key-based (``api_key``) or credential-based
    (``credential``). When both are provided, the API key takes precedence.

    When ``stream=True``, the agent simulates streaming by yielding the response
    word-by-word after obtaining the full completion (the underlying endpoint
    does not support native token streaming).

    Examples:
        Simple generation (works on small compute)::

            agent = FoundryTextGenerationAgent(
                endpoint_url="https://my-endpoint.region.inference.ml.azure.com/score",
                api_key="my-endpoint-key",
                name="BioGPTAgent",
                max_new_tokens=100,
            )
            result = await agent.run("The mechanism of action of aspirin is")

        Sampling with temperature (works on small compute)::

            agent = FoundryTextGenerationAgent(
                endpoint_url="https://my-endpoint.region.inference.ml.azure.com/score",
                api_key="my-endpoint-key",
                name="BioGPTAgent",
                max_new_tokens=200,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.6,
            )
            result = await agent.run("COVID-19 is")

        Beam search (requires larger compute)::

            agent = FoundryTextGenerationAgent(
                endpoint_url="https://my-endpoint.region.inference.ml.azure.com/score",
                api_key="my-endpoint-key",
                name="BioGPTAgent",
                max_new_tokens=100,
                num_beams=5,
                early_stopping=True,
            )
            result = await agent.run("Aspirin inhibits")

        Streaming word-by-word::

            agent = FoundryTextGenerationAgent(
                endpoint_url="https://my-endpoint.region.inference.ml.azure.com/score",
                api_key="my-endpoint-key",
                name="BioGPTAgent",
            )
            async for chunk in agent.run("COVID-19 vaccines work by", stream=True):
                print(chunk.text, end="", flush=True)
    """

    def __init__(
        self,
        endpoint_url: str,
        *,
        credential: AsyncTokenCredential | None = None,
        api_key: str | None = None,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        num_beams: int | None = None,
        early_stopping: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a FoundryTextGenerationAgent.

        This mirrors the HuggingFace ``model.generate()`` parameters used in the
        BioGPT library (https://github.com/microsoft/BioGPT).

        Args:
            endpoint_url: The REST scoring URL of the deployed Azure ML Online Endpoint,
                e.g. ``https://<endpoint-name>.<region>.inference.ml.azure.com/score``.
            credential: An async Azure credential (e.g. ``AzureCliCredential``).
                Used when ``api_key`` is not provided.
            api_key: The endpoint key for key-based authentication. When set, this
                takes precedence over ``credential``.
            max_new_tokens: Maximum number of new tokens to generate. When ``None``
                (default), not sent to the endpoint.
            min_new_tokens: Minimum number of new tokens to generate. When ``None``
                (default), not sent to the endpoint.
            do_sample: Whether to use sampling. When False, uses greedy decoding.
                When ``None`` (default), not sent to the endpoint.
            temperature: Sampling temperature (0.0-1.0). Higher values produce more
                random output. Only used when ``do_sample=True``. When ``None``
                (default), not sent to the endpoint.
            top_k: Number of highest-probability vocabulary tokens to keep for
                top-k filtering. Only used when ``do_sample=True``. When ``None``
                (default), not sent to the endpoint.
            top_p: Cumulative probability for nucleus (top-p) sampling (0.0-1.0).
                Only used when ``do_sample=True``. When ``None`` (default), not
                sent to the endpoint.
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty,
                >1.0 = penalize repetition). When ``None`` (default), not sent
                to the endpoint.
            num_beams: Number of beams for beam search. Requires larger compute
                (e.g., Standard_DS3_v2 or GPU). When ``None`` (default), not sent
                to the endpoint.
            early_stopping: Whether to stop beam search when at least ``num_beams``
                sentences are finished. Requires larger compute. When ``None``
                (default), not sent to the endpoint.
            **kwargs: Additional arguments forwarded to ``BaseAgent``, e.g. ``name``,
                ``description``.
        """
        super().__init__(**kwargs)
        self._endpoint_url = endpoint_url
        self._credential = credential
        self._api_key = api_key

        # Build generation parameters matching HuggingFace transformers API.
        # Only include parameters that are explicitly provided (not None).
        self._generation_params: dict[str, Any] = {}
        if max_new_tokens is not None:
            self._generation_params["max_new_tokens"] = max_new_tokens
        if min_new_tokens is not None:
            self._generation_params["min_new_tokens"] = min_new_tokens
        if do_sample is not None:
            self._generation_params["do_sample"] = do_sample
        if temperature is not None:
            self._generation_params["temperature"] = temperature
        if top_k is not None:
            self._generation_params["top_k"] = top_k
        if top_p is not None:
            self._generation_params["top_p"] = top_p
        if repetition_penalty is not None:
            self._generation_params["repetition_penalty"] = repetition_penalty
        if num_beams is not None:
            self._generation_params["num_beams"] = num_beams
        if early_stopping is not None:
            self._generation_params["early_stopping"] = early_stopping

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

        The agent sends the prompt with HuggingFace generation parameters
        (``max_new_tokens``, ``temperature``, ``top_k``, ``top_p``, etc.) to
        the endpoint in a single call, matching the ``model.generate()`` behavior
        from the HuggingFace ``transformers`` library.

        Args:
            messages: The input prompt. For text-generation, the last user message
                (or the raw string) is used directly as the text prompt.
            stream: If ``True``, return an async iterable that simulates streaming
                by yielding the response word-by-word. If ``False`` (default),
                return an awaitable ``AgentResponse`` with the full generated text.
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
        extracts the last user message as the text prompt to complete.

        Args:
            messages: Normalized list of ``Message`` objects.

        Returns:
            A single string prompt to send to the text-generation endpoint.
        """
        # Find the last user message to use as the prompt.
        for msg in reversed(messages):
            if msg.role == "user" and msg.text:
                return msg.text

        return messages[-1].text if messages else ""

    async def _call_endpoint(self, prompt: str, *, include_params: bool = True) -> str:
        """Call the Azure ML Online Endpoint and return the generated text.

        Sends the prompt along with HuggingFace generation parameters (matching
        the ``model.generate()`` API from the ``transformers`` library) to produce
        a complete generation in a single endpoint call.

        Args:
            prompt: The text prompt to send to the model.
            include_params: Whether to include generation parameters in the
                payload. When ``False``, sends only ``{"inputs": "..."}`` and
                lets the model run with its default settings until it reaches
                its natural token limit. Defaults to ``True``.

        Returns:
            The generated text (which includes the original prompt followed
            by the generated continuation).

        Raises:
            aiohttp.ClientResponseError: When the endpoint returns a non-2xx status.
        """
        headers = await self._get_auth_headers()

        # HuggingFace Inference API payload.
        # When include_params is True and generation params exist,
        # mirrors: model.generate(**inputs, max_new_tokens=..., temperature=..., ...)
        # Otherwise, sends only the input and lets the model generate with
        # its default settings until it reaches its token limit.
        payload: dict[str, Any] = {"inputs": prompt}
        if include_params and self._generation_params:
            payload["parameters"] = self._generation_params

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
        """Non-streaming implementation: single endpoint call with generation parameters."""
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
            This implementation obtains the full completion in a single call
            (without generation parameters to use model defaults), then yields
            it word-by-word to simulate streaming.
        """
        normalized = normalize_messages(messages)
        prompt = self._build_prompt(normalized)

        # Send only {"inputs": "..."} without generation parameters so the
        # model runs with its default settings until it reaches its token limit.
        generated_text = await self._call_endpoint(prompt, include_params=False)

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

    This example shows both small compute and large compute scenarios.
    Small compute (e.g., Standard_B1s) supports basic parameters like max_new_tokens.
    Large compute (e.g., Standard_DS3_v2) supports advanced features like beam search.

    This example uses Azure CLI credential (az login). To use an endpoint key
    instead, set AZURE_ML_API_KEY in your environment and pass it as ``api_key``.
    """
    print("=== Azure AI Foundry - HuggingFace Text Generation Agent Example ===\n")

    # Read configuration from environment variables.
    endpoint_url = os.environ["AZURE_ML_ENDPOINT_URL"]
    api_key = os.environ.get("AZURE_ML_API_KEY")  # Optional; falls back to Azure CLI credential.

    # --- Example 1: Small Compute - Simple parameters ---
    print("--- Example 1: Small Compute (basic parameters) ---")
    print("Suitable for: Standard_B1s, Standard_B2s, or any small SKU\n")

    if api_key:
        # Key-based auth: no credential context manager needed.
        agent = FoundryTextGenerationAgent(
            endpoint_url=endpoint_url,
            api_key=api_key,
            name="BioGPTAgent",
            description="Biomedical text generation agent powered by BioGPT-Large.",
            max_new_tokens=50,
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
                max_new_tokens=50,
            )
            prompt = "Aspirin inhibits the synthesis of prostaglandins by"
            print(f"Prompt: {prompt}")
            result = await agent.run(prompt)
            print(f"Generated: {result.messages[0].text}\n")

    # --- Example 2: Small Compute - Sampling with temperature ---
    print("--- Example 2: Small Compute (sampling) ---")
    print("Suitable for: Standard_B1s, Standard_B2s, or any small SKU\n")

    if api_key:
        agent = FoundryTextGenerationAgent(
            endpoint_url=endpoint_url,
            api_key=api_key,
            name="BioGPTAgent",
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        prompt = "The BRCA1 gene plays a critical role in"
        print(f"Prompt: {prompt}")
        result = await agent.run(prompt)
        print(f"Generated: {result.messages[0].text}\n")
    else:
        async with AzureCliCredential() as credential:
            agent = FoundryTextGenerationAgent(
                endpoint_url=endpoint_url,
                credential=credential,
                name="BioGPTAgent",
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            prompt = "The BRCA1 gene plays a critical role in"
            print(f"Prompt: {prompt}")
            result = await agent.run(prompt)
            print(f"Generated: {result.messages[0].text}\n")

    # --- Example 3: Large Compute - Beam search ---
    print("--- Example 3: Large Compute (beam search) ---")
    print("Requires: Standard_DS3_v2, GPU SKUs, or larger compute\n")
    print("Note: If running on small compute, this example may fail with 424 error.\n")

    if api_key:
        agent = FoundryTextGenerationAgent(
            endpoint_url=endpoint_url,
            api_key=api_key,
            name="BioGPTAgent",
            max_new_tokens=100,
            num_beams=5,
            early_stopping=True,
        )
        prompt = "COVID-19 vaccines work by"
        print(f"Prompt: {prompt}")
        try:
            result = await agent.run(prompt)
            print(f"Generated: {result.messages[0].text}\n")
        except Exception as e:
            print(f"Error (expected on small compute): {e}\n")
    else:
        async with AzureCliCredential() as credential:
            agent = FoundryTextGenerationAgent(
                endpoint_url=endpoint_url,
                credential=credential,
                name="BioGPTAgent",
                max_new_tokens=100,
                num_beams=5,
                early_stopping=True,
            )
            prompt = "COVID-19 vaccines work by"
            print(f"Prompt: {prompt}")
            try:
                result = await agent.run(prompt)
                print(f"Generated: {result.messages[0].text}\n")
            except Exception as e:
                print(f"Error (expected on small compute): {e}\n")

    # --- Example 4: Streaming (works on all compute sizes) ---
    print("--- Example 4: Streaming (no parameters, works on all compute) ---")

    # Streaming sends only {"inputs": "..."} (no generation parameters)
    # so the model runs with its default settings until its token limit.
    if api_key:
        agent = FoundryTextGenerationAgent(
            endpoint_url=endpoint_url,
            api_key=api_key,
            name="BioGPTAgent",
        )
        prompt = "Diabetes mellitus is characterized by"
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
            )
            prompt = "Diabetes mellitus is characterized by"
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

--- Example 1: Small Compute (basic parameters) ---
Suitable for: Standard_B1s, Standard_B2s, or any small SKU

Prompt: Aspirin inhibits the synthesis of prostaglandins by
Generated: Aspirin inhibits the synthesis of prostaglandins by irreversibly
inhibiting cyclooxygenase (COX) enzymes, specifically COX-1 and COX-2.

--- Example 2: Small Compute (sampling) ---
Suitable for: Standard_B1s, Standard_B2s, or any small SKU

Prompt: The BRCA1 gene plays a critical role in
Generated: The BRCA1 gene plays a critical role in DNA damage repair and
genomic stability, with mutations significantly increasing breast cancer risk.

--- Example 3: Large Compute (beam search) ---
Requires: Standard_DS3_v2, GPU SKUs, or larger compute

Prompt: COVID-19 vaccines work by
Generated: COVID-19 vaccines work by training the immune system to recognize
and fight the SARS-CoV-2 virus. They introduce a harmless piece of the virus,
such as the spike protein, prompting the body to produce antibodies and memory
cells that provide protection against future infection.

--- Example 4: Streaming (no parameters, works on all compute) ---
Prompt: Diabetes mellitus is characterized by
Generated: Diabetes mellitus is characterized by elevated blood glucose levels
due to insulin deficiency or resistance.
"""
