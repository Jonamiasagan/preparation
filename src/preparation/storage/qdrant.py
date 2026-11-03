from qdrant_client.grpc import collections_pb2
from qdrant_client import QdrantClient,models
from preparation.config import settings
import uuid

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
    def upsert_chunks(self, chunks: list, vectors: list[list[float]], tenant_id:str | None= None, collection_name:str | None=None ) -> int:
        target_collection = collection_name or settings.collection_name
        points = []
        for chunk,vector in zip(chunks,vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,chunk.chunk_id))
            point = models.PointStruct(
                id = point_id,
                vector= vector,
                payload= {
                    "tenant_id":tenant_id,
                    "text":chunk.text,
                    "header":getattr(chunk,"header",None),
                    "level":getattr(chunk,"level",None),
                    "start_char":getattr(chunk,"start_char",None),
                    "end_char":getattr(chunk,"end_char",None)
                },
            )
            points.append(point)
        self.client.upsert(
            collection_name=target_collection,
            points=points,
        )
        return len(points)
    def search(self, query_vector:list[float],limit:int = 3, tenant_id:str | None=None,collection_name:str | None=None):
        target_collection = collection_name or settings.collection_name
# to filter the multi tenent  
        query_filter= None
        if tenant_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id),
                    )
                ]
            )
        
        results = self.client.query_points(
            collection_name=target_collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return results.points

