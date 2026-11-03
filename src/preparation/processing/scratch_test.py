from preparation.processing.chunker import SlidingWindowChunker
from preparation.processing.embedder import LocalEmbedder

text ="AI Engineering is growing fast. Python is the primary language. Qdrant is our vector store."

chunker =SlidingWindowChunker()
chunking= chunker.chunk_text(text)
texts=[c.text for c in chunking]

embedder = LocalEmbedder()
embedding= embedder.embed_chunks(texts)
print (embedding)