from pydantic import BaseModel, Field
import uuid
class IngestionRequest(BaseModel):
    doc_type: str = "text"
    document_id:str
    tenant_id: str = "default_tenant"
    context: str = Field(...,min_length=10,description='Raw text documents')
    chunk_size: int = Field(default=300, gt=0,description="Size of each chunk in characters")
    chunk_overlap:int = Field(default=40,ge=0,description="Charater overlap between chunks")
class IngestionResponse(BaseModel):
    document_id:str
    tenant_id:str
    total_chunks:int
    collection:str
    status:str ="success"
class search(BaseModel):
    query:str = Field(...,min_length=2,description="query")
    tenant_id:str | None= Field(default = None , description = "query")
    limit:int = Field(default=3,gt=0,le=20,description="results to retrive")
class ResultItem(BaseModel):
    id:str
    score:float
    text:str
    tenant_id: str | None=None
    header:str | None=None
class searchresponse(BaseModel):
    query: str
    total_found : int
    results:list[ResultItem]
