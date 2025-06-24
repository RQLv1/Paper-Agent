import os
from typing import Type, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from create_database import Create_db
from clean_database import Clean_db
from vector_store import VectorStore
from data_extract import DataExtractor
from data_count import DataCount
from data_analysis import FeatureExtractor
from model_sel import ModelSelector
from shapplot import Shapplot

spg_api_key = "" #springer api
els_api_key = "" #elseiver api
wly_api_key = "" #wiley api

llms_api_key = "" #database create api

base_url=""#database create url

abs_prompt_file = r"prompts/abstract_prompt.txt"
exp_prompt_file = r"prompts/experiment_prompt.txt"
titleclean_prompt_file = r"prompts/title_prompt.txt"
wlyclean_prompt_file = r"prompts/wlyclean_prompt.txt"
output_path = r"outputs"

cleanedprop_prompt_file= r"prompts/cleanprop_prompt.txt"
cleanedoper_prompt_file= r"prompts/cleanoper_prompt.txt"


cdb = Create_db(
    spg_api_key= spg_api_key,
    els_api_key = els_api_key,
    wly_api_key = wly_api_key, 
    llms_model = "qwen-plus",
    llms_api_key = llms_api_key,
    base_url= base_url,
    abs_prompt_file = abs_prompt_file, 
    exp_prompt_file = exp_prompt_file,
    titleclean_prompt_file= titleclean_prompt_file,
    wlyclean_prompt_file= wlyclean_prompt_file,
    output_path = output_path,
    agent_model = "ollama_model",
    agent_key = "ollama",
    agent_url= "http://localhost:6006/v1/",
    is_agent=False
    )

cleandb = Clean_db(
    llms_model = "qwen-plus",
    llms_api_key= llms_api_key,
    base_url= base_url,
    cleanedprop_prompt_file= cleanedoper_prompt_file, 
    cleanedoper_prompt_file= cleanedoper_prompt_file,
    output_path=output_path
    )

vectorstore = VectorStore(emb_model= "text-embedding-v3", emb_api_key=llms_api_key, emb_url=base_url)

dataext = DataExtractor(input_path="outputs/dataset.jsonl", output_path= "outputs")
datacount = DataCount()
feaextract = FeatureExtractor()
ms = ModelSelector()
shapplot = Shapplot()

#Model set
class GetDOIsInput(BaseModel):
    query: str = Field(
        description="query keywords for the scopus database."
    )

class GetDOIsClean(BaseModel):
    pass

class StreamModeInput(BaseModel):
    pass

class GetOriginaltext(BaseModel):
    pass

class GetExtracttext(BaseModel):
    pass

class CleanDBInput(BaseModel):
    pass

class FileRagInput(BaseModel):
    query: str = Field(
        description="file rag."
    )
    
    file_lst: List[str] = Field(
        description="multi file path for the pdf/json/txt."
    )

class  Query(BaseModel):
    query: str = Field(
        description="file rag."
    )
    
class Dataext(BaseModel):
    pass

class Datacount(BaseModel):
    pass

class Feaext(BaseModel):
    pass

class ModelSel(BaseModel):
    pass

class Shapplot(BaseModel):
    pass

#Tool set
class GetDOIsTool(BaseTool):
    name: str = Field(default="cdb.get_dois", description="Quickly get the doi of the literature.")
    args_schema : Type[BaseModel] = Field(default=GetDOIsInput)
    
    def _run(self, query: str):
        return cdb.get_dois(query) 

class GetDOIsCleanTool(BaseTool):
    name: str = Field(default="cdb.doi_clean", description="Quickly clean the doi of the literature.")
    args_schema : Type[BaseModel] = Field(default=GetDOIsClean)
    
    def _run(self):
        return cdb.doi_clean()

class GetOriginaltextTool(BaseTool):
    name: str = Field(default= "cdb.get_originaltext", description="Get the original text of the literature.")
    args_schema : Type[BaseModel] = Field(default=GetOriginaltext)
    
    def _run(self):
        return cdb.get_originaltext()

class GetExtracttextTool(BaseTool):
    name: str = Field(default= "cdb.get_ext", description="Get the extracted text of the literature.")
    args_schema : Type[BaseModel] = Field(default=GetExtracttext)
    
    def _run(self):
        return cdb.get_ext()
    
class StreamModeTool(BaseTool):
    name: str = Field(default= "cdb.stream_mode", description= "create the database.")
    args_schema : Type[BaseModel] = Field(default=StreamModeInput)

    def _run(self):
        return cdb.stream_mode()

class CleanDBTool(BaseTool):
    name: str = Field(default= "cleandb.clean_db", description= "Clean the database.")
    args_schema : Type[BaseModel] = Field(default=CleanDBInput)

    def _run(self):
        return cleandb.clean_db()
    
class FileRagTool(BaseTool):
    name: str = Field(default= "vectorstore.rag_query", 
                     description= "File quiz tool. analysis file")
    args_schema : Type[BaseModel] = Field(default=FileRagInput)
    def _run(self, file_lst, query):
        if file_lst:
            for file in file_lst:
                if not os.path.exists(file):
                    raise ValueError(f"file not exist: {file}")
                vectorstore.create_vectorstore([file])
        return vectorstore.query_vectorstore(query)
    
class QueryTool(BaseTool):
    name: str = Field(default= "vectorstore.query", description= "File quiz tool. Use when you need to quiz based on uploaded files, no need to upload files again for subsequent queries.")
    args_schema : Type[BaseModel] = Field(default=Query)

    def _run(self, query):
        return vectorstore.query_vectorstore(query)
    
class DataextTool(BaseTool):
    name: str = Field(default= "dataext.exe", description= "Extract the jsonl to csv.")
    args_schema : Type[BaseModel] = Field(default=Dataext)

    def _run(self):
        return dataext.exe()

class DatacountTool(BaseTool):
    name: str = Field(default= "datacount.exe", description= "Count the data.")
    args_schema : Type[BaseModel] = Field(default=Datacount)

    def _run(self):
        return datacount.exe()

class FeaextTool(BaseTool):
    name: str = Field(default= "feaextract.exe", description= "Extract the feature.")
    args_schema : Type[BaseModel] = Field(default=Feaext)

    def _run(self):
        return feaextract.exe()

class ModelSelTool(BaseTool):
    name: str = Field(default= "ms.exe", description= "Select the model.")
    args_schema : Type[BaseModel] = Field(default=ModelSel)

    def _run(self):
        return ms.exe()
    
class ShapplotTool(BaseTool):
    name: str = Field(default= "shapplot.shap_plot", description= "Plot the shap.")
    args_schema : Type[BaseModel] = Field(default=Shapplot)

    def _run(self):
        return shapplot.shap_plot()








