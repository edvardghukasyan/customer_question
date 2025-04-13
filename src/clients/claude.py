from src.clients.base import BaseClient


class ClaudeClient(BaseClient):
    def invoke(self, prompt: str, **kwargs):
        raise NotImplementedError("Claude client is not implemented yet.")
    