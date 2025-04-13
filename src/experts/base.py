from abc import ABC, abstractmethod


class BaseExpert(ABC):
    
    @abstractmethod
    async def run(self, data):
        pass
