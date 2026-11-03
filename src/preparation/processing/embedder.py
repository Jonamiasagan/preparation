from fastembed import TextEmbedding
from preparation.config import settings

class LocalEmbedder:
    def __init__(self,model_name:str | None = None):
        self.model_name=model_name or settings.embedding_model
        self.model = TextEmbedding(model_name=self.model_name)
    def embed_chunks(self, texts:list[str]) -> list[list[float]]:
        if len(texts) == 0 : return []
        embeddings = self.model.embed(texts)
        list_of_floats = [vector.tolist() for vector in embeddings]
        return list_of_floats   
    def embed_query(self, query:str) -> list[float] :
        if not query.strip():
            raise ValueError("Query cannot be empty")
        return self.embed_chunks([query])[0]

