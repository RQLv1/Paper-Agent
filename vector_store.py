import torch
import re
import os
from typing import List
from nltk.tokenize import sent_tokenize
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document
from chromadb.config import Settings

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from tqdm import tqdm
from typing import List
import os, shutil
from openai import OpenAI
from langchain.schema.embeddings import Embeddings

class ScientificDocumentSplitter:
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        sentence_safe: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_safe = sentence_safe
        

    def _process_pdf(self, file_path):
        
        pdf_file_name = file_path
        name_without_suff = pdf_file_name.split(".")[0]
        local_image_dir, local_md_dir = "output/images", "output"
        image_dir = str(os.path.basename(local_image_dir))
        os.makedirs(local_image_dir, exist_ok=True)

        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
            local_md_dir
        )

        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

        ds = PymuDocDataset(pdf_bytes)

        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        md_content = pipe_result.get_markdown(image_dir)

        paragraphs = re.split(r"\n\s*\n", md_content)
        chunks = []
        for para_num, para in enumerate(paragraphs[8:-8]):
            if para.strip():
                meta = {
                    "source": file_path,
                    "file_type": "txt",
                    "paragraph": para_num + 1
                }
                chunks.extend(self._split_text(para, meta))

        return chunks

    def _process_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        for para_num, para in enumerate(paragraphs):
            if para.strip():
                meta = {
                    "source": file_path,
                    "file_type": "txt",
                    "paragraph": para_num + 1
                }
                chunks.extend(self._split_text(para, meta))
        return chunks

    def _process_json(self, file_path):
        
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".", 
            text_content=False,                
            json_lines=True    
        )
        
        
        langchain_docs = loader.load_and_split()

        
        return langchain_docs 
    
    def _split_text(self, text, metadata):
        if self.sentence_safe:
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                new_length = current_length + sentence_length + (1 if current_chunk else 0)
                
                if new_length > self.chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "page_content": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_len": len(chunk_text),
                            "method": "sentence_safe",
                            "start_sentence": len(chunks)*self.chunk_overlap
                        }
                    })
                    
                    overlap_remaining = self.chunk_overlap
                    overlap_sentences = []
                    for s in reversed(current_chunk):
                        if overlap_remaining <= 0:
                            break
                        overlap_sentences.insert(0, s)
                        overlap_remaining -= len(s) + 1  # 1 for space
                    
                    current_chunk = overlap_sentences
                    current_length = sum(len(s) for s in current_chunk) + max(0, len(current_chunk)-1)
                
                current_chunk.append(sentence)
                current_length = new_length
            
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "page_content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_len": len(chunk_text),
                        "method": "sentence_safe"
                    }
                })
            return chunks
        else:
            text = text.replace("\n", " ")
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                
                if end < len(text):
                    last_punct = max(chunk.rfind("."), chunk.rfind("?"), chunk.rfind("!"))
                    if last_punct > 0.8 * self.chunk_size:
                        end = start + last_punct + 1
                        chunk = text[start:end]
                
                chunks.append({
                    "page_content": chunk.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_len": len(chunk),
                        "method": "sliding_window"
                    }
                })
                start = end - self.chunk_overlap
            return chunks

    def _calculate_overlap_sentences(self, sentences: List[str]) -> int:
        
        return self._num_sentences_for_overlap()

    def _num_sentences_for_overlap(self) -> int:
        
        if self.chunk_size == 0: return 1
      
        num_sentences = int((self.chunk_overlap / self.chunk_size) * 5) 
        return max(1, num_sentences) if self.chunk_overlap > 0 else 0

    def process_file(self, file):
        ext = os.path.splitext(file)[1].lower()
        
        if file.endswith(".pdf"):
            return self._process_pdf(file)
        elif file.endswith(".txt"):
            return self._process_txt(file)
        elif file.endswith(".jsonl") or file.endswith(".json"):
            return self._process_json(file)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

class DashScopeEmbeddings(Embeddings):
    def __init__(self, api_key, base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1", model = "text-embedding-v3", dimensions: int = 1024):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs using DashScope API"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text using DashScope API"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding

class VectorStore:
    def __init__(self,
                 emb_model,
                 emb_api_key,
                 emb_url,
                 splitter_chunk_size: int = 1000,
                 splitter_chunk_overlap: int = 300,
                 splitter_sentence_safe: bool = True,
                 embedding_dimensions: int = 1024):
        
        self.emb_api_key = emb_api_key

        self.client_settings = Settings(
            is_persistent=True,
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        )

        self.splitter = ScientificDocumentSplitter(
            chunk_size=splitter_chunk_size,
            chunk_overlap=splitter_chunk_overlap,
            sentence_safe=splitter_sentence_safe,
        )

        # 创建DashScope嵌入模型实例
        self.embed_model = DashScopeEmbeddings(
            api_key=emb_api_key,
            base_url=emb_url,
            model=emb_model,
            dimensions=embedding_dimensions
        )

        safe_model_name = emb_model
        self.collection_name = f"rag_sci_collection_{safe_model_name}"
        print(f"VectorStore initialized. Chroma collection name: {self.collection_name}")

    def create_vectorstore(self, file_lst: list):
        if os.path.exists("./chroma_db"):
            print("Removing existing Chroma database...")
            shutil.rmtree("./chroma_db")
            
        merged_docs = []
        print(f"Processing {len(file_lst)} file(s)...")
        for file_path in tqdm(file_lst):
            try:
                processed_output = self.splitter.process_file(file_path)
                if isinstance(processed_output, list) and processed_output:
                    if isinstance(processed_output[0], Document):
                        docs_to_add = processed_output
                    elif isinstance(processed_output[0], dict):
                        docs_to_add = [Document(page_content=chunk['page_content'], 
                                      metadata=chunk['metadata']) for chunk in processed_output]
                    else:
                        print(f"Warning: Unknown output type from splitter for file {file_path}. Skipping.")
                        docs_to_add = []
                    merged_docs.extend(docs_to_add)
                elif not processed_output:
                     print(f"Warning: No content processed from file {file_path}.")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
            
        if not os.path.exists("./chroma_db"):
            print(f"Creating Chroma persist directory: ./chroma_db")
            os.makedirs("./chroma_db")
            
        print(f"Creating/loading vector store for collection: {self.collection_name}...")
        vectorstore = Chroma.from_documents(
            documents=merged_docs,
            embedding=self.embed_model,
            collection_name=self.collection_name,
            client_settings=self.client_settings,
            persist_directory="./chroma_db"
        )
        vectorstore.persist()

        return vectorstore

    def query_vectorstore(self, query: str):
        
        if os.path.exists("./chroma_db"):
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embed_model,
                client_settings=self.client_settings,
                persist_directory="./chroma_db"
            )

            print(f"Searching for query: '{query}'")
            retrieved_docs = vectorstore.max_marginal_relevance_search(
                query,
                k=10,
                fetch_k=500,
                lambda_mult=0.7
            )

            if not retrieved_docs:
                source_knowledge = "No relevant contexts found for the query."
            else:
                source_knowledge_parts = []
                for i, doc in enumerate(retrieved_docs):
                    source_knowledge_parts.append(f"{doc.page_content}")
                source_knowledge = "\n\n---\n\n".join(source_knowledge_parts)

            augment_prompt = f"""
            Based on the following extracted contexts, please answer the query.

            Contexts:
            {source_knowledge}

            Query: {query}
            Answer:
            """
        else:
            augment_prompt = query
            
        return augment_prompt