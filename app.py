import gradio as gr
import datetime, os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import warnings
warnings.filterwarnings("ignore")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['GRADIO_TEMP_DIR'] = APP_DIR
def create_tools(spg_key, 
                 els_key, 
                 wly_key,
                 data_analysis_model,
                 data_analysis_key, 
                 data_analysis_url,
                 emb_model,
                 emb_key, 
                 emb_url,
                 is_agent = False,
                 agent_model = "ollama_model",
                 agent_key = "ollama",
                 agent_url = "http://localhost:6006/v1/",
                 sel_prop = "refractive index", lb=1.0, ub=2.5):
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

    absolute_output_path = os.path.join(APP_DIR, "outputs")
    abs_prompt_file = os.path.join(APP_DIR, "prompts/abstract_prompt.txt")
    exp_prompt_file = os.path.join(APP_DIR, "prompts/experiment_prompt.txt")
    titleclean_prompt_file = os.path.join(APP_DIR, "prompts/title_prompt.txt")
    wlyclean_prompt_file = os.path.join(APP_DIR, "prompts/wlyclean_prompt.txt")
    cleanedprop_prompt_file=os.path.join(APP_DIR, "prompts/cleanprop_prompt.txt")
    cleanedoper_prompt_file=os.path.join(APP_DIR, "prompts/cleanoper_prompt.txt")
    dataset_file = os.path.join(APP_DIR, "outputs/dataset.jsonl")

    cdb = Create_db(
        spg_api_key=spg_key,
        els_api_key=els_key,
        wly_api_key=wly_key,
        llms_model= data_analysis_model,
        llms_api_key= data_analysis_key,
        base_url= data_analysis_url,
        abs_prompt_file=abs_prompt_file,
        exp_prompt_file=exp_prompt_file,
        titleclean_prompt_file= titleclean_prompt_file,
        wlyclean_prompt_file= wlyclean_prompt_file,
        output_path= absolute_output_path,
        agent_model = agent_model,
        agent_key = agent_key,
        agent_url= agent_url,
        is_agent= is_agent,
    )
    cleandb = Clean_db(
        llms_model = data_analysis_model,
        llms_api_key=data_analysis_key,
        base_url=data_analysis_url,
        cleanedprop_prompt_file=cleanedprop_prompt_file,
        cleanedoper_prompt_file=cleanedoper_prompt_file,
        output_path=absolute_output_path
    )
    vectorstore = VectorStore(emb_model= emb_model, emb_api_key=emb_key, emb_url= emb_url)
    dataext = DataExtractor(input_path=dataset_file, output_path= absolute_output_path)
    datacount = DataCount()
    feaextract = FeatureExtractor(sel_prop=sel_prop, lb=lb, ub=ub)
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
    
    return [
        GetDOIsTool(description="Quickly get the doi of the literature."),
        GetDOIsCleanTool(description="Quickly clean the doi of the literature."),
        GetOriginaltextTool(description="Get the original text of the literature."),
        GetExtracttextTool(description="Get the extracted text of the literature."),
        StreamModeTool(description="create the database."),
        CleanDBTool(description="Clean the database."),
        FileRagTool(description="File quiz tool. analysis file."),
        QueryTool(description="File quiz tool. Use when you need to quiz based on uploaded files, no need to upload files again for subsequent queries."),
        DataextTool(description="Extract the jsonl to csv."),
        DatacountTool(description="Count the data."),
        FeaextTool(description="Extract the feature."),
        ModelSelTool(description="Select the model."),
        ShapplotTool(description="Plot the shap.")
    ]

def generate_outputs_tree():
    output_dir = os.path.join(APP_DIR, "outputs")
    if not os.path.exists(output_dir):
        return "outputs directory not found"
    
    tree = []
    for root, dirs, files in os.walk(output_dir):
        level = root[len(output_dir)+1:].count(os.sep)
        indent = '│   ' * level + '├── '
        tree.append(f"{indent}{os.path.basename(root)}/")
        for f in files:
            file_indent = '│   ' * (level+1) + '├── '
            tree.append(f"{file_indent}{f}")
    return '\n'.join(tree)

memory = MemorySaver()

with open("prompts/calling_prompts_gradio.txt", "r", encoding=  "utf-8") as f:
    prompts = f.read()

with gr.Blocks() as demo:
    gr.Markdown("## Paper analyzer")
    with gr.Accordion("📖 Table of Contents", open=False):
        toc_image = gr.Image(
            value=os.path.join(APP_DIR, "ToC.png"),
            label="System Architecture",
            interactive=False,
            show_download_button=False
        )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):  
            with gr.Accordion("📁 Real-time file directory", open=True):
                dir_display = gr.Textbox(
                    label="Outputs Directory",
                    value=generate_outputs_tree,
                    every=2,
                    lines=22,
                    interactive=False
                )
        
        with gr.Column(scale=4):
            with gr.Row():
                input_box = gr.Textbox(label="Input", placeholder="Enter your paper analysis needs...")
                file_uploader = gr.File(label="Upload file (optional)", 
                                        file_count="multiple",
                                        interactive=True)
                submit_btn = gr.Button("Submit")
            
            output_box = gr.Chatbot(
                label="Records of dialogues", 
                height=500,
                show_label=True,
                container=True,
                avatar_images=("user.png", "bot.png")
            )
    
    with gr.Accordion("🔑 API Key Configuration", open=False):
        with gr.Tabs():
            with gr.Tab("Paper platform"):
                gr.Markdown("### Paper database API")
                with gr.Row():
                    spg_key = gr.Textbox(
                        label="Springer Key",
                        type="password",
                        value=""
                    )
                    els_key = gr.Textbox(
                        label="Elsevier Key",
                        type="password",
                        value=""
                    )
                    wly_key = gr.Textbox(
                        label="Wiley Key",
                        type="password",
                        value=""
                    )
                    data_analysis_model = gr.Textbox(
                        label="Data analysis model",
                        value="qwen-plus"
                    )
                    data_analysis_key = gr.Textbox(
                        label="Data analysis Key",
                        type="password",
                        value=""
                    )
                data_analysis_url = gr.Textbox(
                    label="Data analysis url",
                    value=""
                )
                with gr.Tab("Agent"):
                    gr.Markdown("### try agent to extract the data")
                    is_agent = gr.Checkbox(
                        label="Use agent",
                        value=False,
                        info="set agent parameters"
                    )

                    with gr.Column(visible=False) as agent_config:
                        with gr.Row():
                            agent_model = gr.Textbox(
                                label="Agent model",
                                value="ollama_model"
                            )
                            agent_key = gr.Textbox(
                                label="Agent API Key",
                                type="password",
                                value="ollama"
                            )
                        agent_url = gr.Textbox(
                                label="Agent API url",
                                value="http://localhost:6006/v1/"
                            )
                
                    is_agent.change(
                        fn=lambda x: gr.Column(visible=x),
                        inputs=is_agent,
                        outputs=agent_config
                    )

            with gr.Tab("Dialog model"):
                gr.Markdown("### Dialog LLMs service configuration")
                with gr.Row():
                    model_name = gr.Textbox(
                    label="Model selection",
                    value="qwen-plus"
                    )
                    llm_key = gr.Textbox(
                        label="LLM API Key",
                        type="password",
                        value=""
                    )
                llm_url = gr.Textbox(
                    label="API url",
                    value=""
                )
                

            with gr.Tab("Embedding model"):
                gr.Markdown("### Use for file analysis")
                with gr.Row():
                    emb_model = gr.Textbox(
                        label="Embedding model",
                        value="text-embedding-v3"
                    )
                    emb_key = gr.Textbox(
                        label="Embedding API Key",
                        type="password",
                        value=""
                    )
                emb_url = gr.Textbox(
                    label="Embedding base url",
                    value=""
                )

    with gr.Accordion("🔍 Data filtering parameters", open=False):
            sel_prop = gr.Textbox(
                label="Filter Properties", 
                value="refractive index",
                placeholder="Enter the name of the property to be filtered"
            )
            lb_input = gr.Number(
                label="lower bound", 
                value=1.2,
            )
            ub_input = gr.Number(
                label="upper bound",
                value=2.5,
            )
    def stream_response(message, history, uploaded_file_objects, 
                    spg_key, 
                    els_key, 
                    wly_key, 
                    data_analysis_model, 
                    data_analysis_key, 
                    data_analysis_url,
                    model_name, 
                    llm_key, 
                    llm_url,
                    emb_model,
                    emb_key, 
                    emb_url,
                    is_agent,
                    agent_model, 
                    agent_key, 
                    agent_url,
                    sel_prop, 
                    lb, 
                    ub):
    
        config = {"configurable": {"thread_id": "abc123"}}
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        actual_file_paths = []
        if uploaded_file_objects:
            for temp_file_obj in uploaded_file_objects:
                actual_file_paths.append(temp_file_obj.name)
            print(f"Gradio temporary path for uploaded files: {actual_file_paths}")
        augmented_message = message
        if actual_file_paths:
            augmented_message += f"\n[System alert: File uploaded: {', '.join(actual_file_paths)}]"

        tools = create_tools(spg_key, 
                             els_key, 
                             wly_key,
                             data_analysis_model,
                             data_analysis_key, 
                             data_analysis_url, 
                             emb_model,
                             emb_key, 
                             emb_url,
                             is_agent,
                             agent_model,
                             agent_key,
                             agent_url,
                             sel_prop, 
                             lb, 
                             ub)
        
        current_model = ChatOpenAI(
            openai_api_key=llm_key,
            base_url=llm_url,
            model=model_name
        )

        agent_executor = create_react_agent(current_model, tools, prompt=prompts, checkpointer=memory)

        full_response = ""
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=augmented_message)]}, config
        ):
            if "agent" in chunk and chunk['agent']['messages'][0].content:
                full_response += chunk['agent']['messages'][0].content + "\n"

        end_time = datetime.datetime.now().strftime("%H:%M:%S")
        time_info = f"⏱️ Response time：{start_time.split()[1]} - {end_time} ({datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time.split()[1], '%H:%M:%S')}"

        history.append((augmented_message, f"{full_response}\n\n{time_info}"))
        return history

    submit_btn.click(
        fn=stream_response,
        inputs=[input_box, output_box, file_uploader, 
                spg_key, els_key, wly_key, 
               data_analysis_model, data_analysis_key, data_analysis_url,
               model_name, llm_key, llm_url, 
               emb_model, emb_key, emb_url,
               is_agent, agent_model, agent_key, agent_url, 
               sel_prop, lb_input, ub_input],
        outputs=output_box
    )

demo.launch()