import os
import json

from ollama import Client
from openai import OpenAI

from config import load_env


class LLMService:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        load_env()

        self.model = model or os.getenv("LLM_MODEL", "llama3.2:3b")
        self.base_url = base_url or os.getenv("BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("API_KEY") or None

        if self.api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = Client(host=self.base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        format: dict | None = None,
    ) -> str:
        """
        Base chat method.

        Accepts a list of messages:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        If format is None, returns a plain text response.
        If format contains a JSON schema, instructs the model to follow it.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list")

        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("Each message must contain 'role' and 'content'")

        try:
            if self.api_key:
                kwargs: dict = {"model": self.model, "messages": messages}

                if format is not None:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_response",
                            "strict": True,
                            "schema": format,
                        },
                    }

                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            # Local Ollama
            kwargs = {"model": self.model, "messages": messages}
            if format is not None:
                kwargs["format"] = format

            response = self.client.chat(**kwargs)
            return response.message.content

        except Exception as e:
            provider = "OpenAI-compatible API" if self.api_key else "Ollama"
            raise RuntimeError(f"{provider} chat failed: {e}") from e

    def generate(self, prompt: str) -> str:
        """
        Convenience wrapper for single-prompt calls with no system message
        and no structured output requirement.

        Example:
            llm.generate("Explain clustering in one sentence.")
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("`prompt` must be a non-empty string")

        return self.chat([{"role": "user", "content": prompt}])

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict,
    ) -> dict:
        """
        Structured-output wrapper.

        Calls chat() with the given schema, then parses the response as JSON
        and returns a Python dict.

        Example:
            result = llm.chat_json(messages, schema)
            result["name"]
        """
        raw_response = self.chat(messages=messages, format=schema)

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model did not return valid JSON. Raw response:\n{raw_response}"
            ) from e


if __name__ == "__main__":
    llm_service = LLMService()
    response = llm_service.generate("What is the capital of France?")
    print(response)