from abc import ABC, abstractmethod


class BaseClient(ABC):
    
    @abstractmethod
    def invoke(self, prompt: str, **kwargs):
        pass
