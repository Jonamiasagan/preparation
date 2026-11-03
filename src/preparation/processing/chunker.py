from pydantic.dataclasses import dataclass
from preparation.config import settings
import uuid
@dataclass  

class chunk:
    chunk_id: str
    text: str
    start_char: int
    end_char: int

class SlidingWindowChunker:
    def __init__(self,chunk_size: int | None=None, chunk_overlap: int | None=None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        if self.chunk_overlap >= self.chunk_size: 
            raise ValueError("Chunk overlap must be smaller than chunk size")
    def chunk_text(self, text: str) -> list[chunk]:
        if not text.strip(): return []
        if len(text)<=self.chunk_size: 
            return [
                chunk(
                    chunk_id= str(uuid.uuid4()),
                    text=text,
                    start_char= 0,
                    end_char=len(text)
            )
        ]
        chunks : list[chunk]=[]
        start = 0
        step = self.chunk_size-self.chunk_overlap
        while start < len(text):
            chunk_text = text[start : start+self.chunk_size]
            end_char = start+len(chunk_text)
            new_chunk = chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                start_char=start,
                end_char=end_char
            )
            chunks.append(new_chunk)
            if end_char >=len(text):
                break
            start += step
        return chunks
    




    