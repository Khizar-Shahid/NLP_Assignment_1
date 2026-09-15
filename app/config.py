"""
Central configuration for the chatbot.

Everything that you might want to tune (which model, how big the context
window is, where Ollama lives) is collected here so the rest of the code never
hard-codes a value. Values can be overridden with environment variables, which
makes it easy to change the model without editing code.
"""
import os


class Settings:
    # --- LLM / Ollama ---
    # The Ollama HTTP endpoint. Ollama listens on 11434 by default.
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # The model tag as it appears in `ollama list`.
    MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:3b-instruct-q4_K_M")
    # Sampling controls. Low temperature keeps the assistant on-policy.
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.4"))
    # Hard cap on tokens the model may generate per reply (keeps latency bounded).
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))

    # --- Context window management ---
    # Approximate token budget for the whole prompt we send to the model.
    # The system prompt is always kept; older turns are dropped once we exceed
    # this budget. See app/conversation.py for the algorithm.
    CONTEXT_TOKEN_BUDGET: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "1800"))
    # We never keep more than this many *messages* of history regardless of size.
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

    # --- Server ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
