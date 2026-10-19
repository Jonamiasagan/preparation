from pydantic_settings import BaseSettings,SettingsConfigDict

class settings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    collection_name: str = "rag_knowledge_base"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_size: int = 500
    chunk_overlap: int = 100
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
settings = settings()
