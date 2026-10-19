from qdrant_client.grpc import collections_pb2
from qdrant_client import QdrantClient,models
from preparation.config import settings

class QdrantStorage:
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host,port=settings.qdrant_port)
    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e :
            return False
    def ensure_collection_exists(self, collection_name: str | None = None) -> bool:
        target_collection = collection_name or settings.collection_name
        if self.client.collection_exists(collection_name=target_collection):
            return False
        else:
            self.client.create_collection(collection_name= target_collection,
            vectors_config=models.VectorParams(size=settings.embedding_dim,
            distance= models.Distance.COSINE))           
            return True
        