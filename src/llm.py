import os
import json
from typing import Optional, List, Dict, Any

from ollama import Client
from openai import OpenAI

from config import load_env


class LLMService:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        load_env()

        self.model = model or os.getenv("LLM_MODEL", "llama3.2:3b")
        self.base_url = base_url or os.getenv("BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("API_KEY") or None

        # Se c'è API_KEY uso client OpenAI-compatible.
        # Altrimenti uso Ollama locale.
        if self.api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = Client(host=self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        format: dict | None = None,
    ) -> str:
        """
        Metodo base.

        Riceve una lista di messaggi:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

        Se format è None:
            risposta testuale normale.

        Se format contiene uno schema JSON:
            chiede al modello di rispettare quello schema.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list")

        for message in messages:
            if (
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
            ):
                raise ValueError("Each message must contain 'role' and 'content'")

        try:
            if self.api_key:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                }

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

            # Ollama locale
            kwargs = {
                "model": self.model,
                "messages": messages,
            }

            if format is not None:
                kwargs["format"] = format

            response = self.client.chat(**kwargs)
            return response.message.content

        except Exception as e:
            provider = "OpenAI-compatible API" if self.api_key else "Ollama"
            raise RuntimeError(f"{provider} chat failed: {e}") from e

    def generate(self, prompt: str) -> str:
        """
        Helper semplice per chiamate single-prompt.

        Lo usiamo quando NON c'è bisogno di separare system/user
        e NON ci serve JSON strutturato.

        Esempio:
            llm.generate("Explain clustering in one sentence.")
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("`prompt` must be a non-empty string")

        messages = [
            {"role": "user", "content": prompt}
        ]

        return self.chat(messages)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        schema: dict,
    ) -> Dict[str, Any]:
        """
        Helper per structured output.

        Si usa quando vogliamo che il modello restituisca JSON valido.

        Flusso:
        1. chiama chat() passando lo schema JSON;
        2. riceve una stringa;
        3. fa json.loads();
        4. restituisce un dizionario Python.

        Esempio:
            result = llm.chat_json(messages, schema)
            result["name"]
        """
        raw_response = self.chat(
            messages=messages,
            format=schema,
        )

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