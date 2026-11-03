from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from app.schemas import IngestionRequest, IngestionResponse,search,ResultItem,searchresponse
from preparation.processing.chunker import SlidingWindowChunker
from preparation.processing.embedder import LocalEmbedder
from preparation.processing.markdown_chunker import MarkdownHeaderChunker
from preparation.storage.qdrant import QdrantStorage
from preparation.config import settings
from pydantic import BaseModel

storage = QdrantStorage()
embedder = LocalEmbedder()

@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.ensure_collection_exists()
    yield
app=FastAPI(title="Production Rag Engine", lifespan=lifespan)

@app.get("/health")
def health_check():
    return{
        "status":"healthy",
        "qdrant_alive":storage.health()
    }
@app.post("/api/v1/documents/ingest",status_code=status.HTTP_201_CREATED,
response_model=IngestionResponse)
def ingest_document(payload: IngestionRequest):
    # chunker = SlidingWindowChunker(
    #     chunk_size=payload.chunk_size,
    #     chunk_overlap=payload.chunk_overlap
    # )
    if payload.doc_type == "markdown":
        chunker = MarkdownHeaderChunker(
            max_chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap
        )
        chunks = chunker.chunk_markdown(payload.context)
    else:
        chunker = SlidingWindowChunker(
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap
        )
        chunks = chunker.chunk_text(payload.context)
    texts = [c.text for c in chunks]
    vectors = embedder.embed_chunks(texts)
    total_stored = storage.upsert_chunks(chunks, vectors, tenant_id=payload.tenant_id)

    return IngestionResponse(
        document_id=payload.document_id,
        tenant_id=payload.tenant_id,
        total_chunks=total_stored,
        collection=settings.collection_name,
        status="success"
    )
@app.post("/api/v1/search", response_model=searchresponse)
def search_documents(payload: search):
    query_vector=embedder.embed_query(payload.query)
    raw_results = storage.search(
        query_vector=query_vector,
        limit=payload.limit,
        tenant_id=payload.tenant_id
    )
    items = []
    for pt in raw_results:
        items.append(
            ResultItem(
                id = str(pt.id),
                score = round(pt.score,4),
                text= pt.payload.get("text",""),
                tenant_id=pt.payload.get("tenant_id"),
                header= pt.payload.get("header")
            )
        )
    return searchresponse(
        query= payload.query,
        total_found=len(items),
        results=items
    )