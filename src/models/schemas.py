from pydantic import BaseModel


class ClusterResult(BaseModel):
    question: str
    cluster_name: str
    description: str
