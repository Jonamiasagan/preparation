import uuid
from pydantic import BaseModel
from preparation.processing.chunker import SlidingWindowChunker


class MarkdownChunk(BaseModel):
    chunk_id: str
    text: str
    header: str
    level: int


class MarkdownHeaderChunker:
    def __init__(self, max_chunk_size: int = 500, chunk_overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback_chunker = SlidingWindowChunker(max_chunk_size, chunk_overlap)

    def chunk_markdown(self, markdown_text: str) -> list[MarkdownChunk]:
        chunks: list[MarkdownChunk] = []
        lines = markdown_text.splitlines()

        current_header = "Introduction"
        current_level = 1
        current_lines: list[str] = []

        def flush_section():
            text_block = "\n".join(current_lines).strip()
            if not text_block:
                return

            if len(text_block) <= self.max_chunk_size:
                chunks.append(
                    MarkdownChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=text_block,
                        header=current_header,
                        level=current_level,
                    )
                )
            else:
                subchunks = self.fallback_chunker.chunk_text(text_block)
                for sc in subchunks:
                    chunks.append(
                        MarkdownChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=sc.text,
                            header=current_header,
                            level=current_level,
                        )
                    )
        # The Line Loop: read line-by-line and dump on #
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                flush_section()
                current_lines.clear()
                hash_count = len(stripped) - len(stripped.lstrip("#"))
                current_level = hash_count
                current_header = stripped.lstrip("#").strip()
            else:
                current_lines.append(line)
        # Final flush at end of file
        flush_section()
        return chunks
