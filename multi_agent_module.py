from autogen import GroupChatManager, GroupChat
import autogen
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent, register_function
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from preprocessing import *
from rag_chains import *
from typing import Annotated
import faiss
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
load_dotenv()


model = "gpt-4-32k-0613" # change it to whichever model
api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ["OPENAI_API_BASE"]
llm_config = {
    "config_list": [{"model":model , "api_key":api_key, "base_url" : base_url, "price" : [0.06, 0.12]}],
    "cache_seed" : 42
}
config_list = llm_config

gpt_4_llm = ChatOpenAI(
    model="gpt-4-0613-openai",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key,  # if you prefer to pass api key in directly instaed of using env vars
    base_url=base_url,
    # organization="...",
    # other params...
)

def make_rag_tool(qa_chain):
    def rag_tool(query: Annotated[str, "User question to retrieve from vectorstore"]) -> str:
        response = qa_chain.invoke({"query" : query})
        result = response['result'].strip()
        if result.lower().startswith("i'm sorry"):
            return "I couldn't find any relevant information in my document database to answer the question. Forwarding to the next agent."
        return result
    return rag_tool
    
# def rag_tool(query: Annotated[str, "User question to retrieve from vectorstore"]) -> str:
#     print(f"[RAG Agent] Checking vector DB for: {query}")
#     response = qa_chain.invoke({"query" : query})
#     result = response["result"].strip()

#     if result.lower().startswith("i'm sorry"):
#         return "I couldn't find any relevant information in my document database to answer the question. Forwarding to the next agent."

#     return result

def collecting_agents(qa_chain):
    planner_agent = AssistantAgent(
        name="planner",
        system_message=""" 
        Planner. Suggest a plan. Revise the plan based on feedback from user, until user approval.
        The plan may involve an engineer who can write code and a reviewer who doesn't write code.
        Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a reviewer.
        """,
        llm_config=llm_config
    )
    tool_func = make_rag_tool(qa_chain)
    rag_agent = AssistantAgent(
        name="RAG_agent",
        system_message="You are a document-based expert that only answers queries using a vector database of documents.",
        llm_config=llm_config,
        code_execution_config = {
        "work_dir" : None,
        "use_docker" : False,
        "enabled" : False
        },
        function_map={
            "retrieve_from_docs" : tool_func
        }
    )

    rag_agent.register_for_llm(name="retrieve_from_docs", description="Check VectorDB for document-based answers")(tool_func)
    # rag_agent.register_for_llm(name="retrieve_from_docs", description="Check VectorDB for document-based answers")(rag_tool)


    continuation_agent = AssistantAgent(
        name="Continuation_Agent",
        system_message=""" 
        You are responsible for reviewing the latest answer given by other agenets.
        If the answer is complete and satifactorily addresses the user's question,
        response with 'TERMINATE' to end the conversation. Otherwise suggest what is missing or unclear
        """,
        llm_config=llm_config
    )

    user_proxy = UserProxyAgent(
        name="user",
        system_message="A human user. Interact with the planner to discuss the plan. Plan execution needs to be approved by this user. First step is to check whether the rag_agent using the qa_chain is able to answer the user query.",
        human_input_mode="TERMINATE",
        code_execution_config = {"use_docker" : False, "last_n_messages" : 3, "work_dir" : "groupchat"}
    )

    return planner_agent, rag_agent, continuation_agent, user_proxy

def add_another_doc_v3(pdf_dir, text_splitter, medembed_embeddings):
    loader = pdfplumber.open(pdf_dir)
    docs = [Document(page_content = s.extract_text(), metadata={"page_num" : i + 1, "source_file" : pdf_dir}) for i, s in enumerate(loader.pages)]
    docs_chunks = text_splitter.split_documents(docs)
    tables = get_all_the_tables(pdf_dir)
    clean_tables = [cleanup_tables(table.df) for table in tables]
    mmd_tables = [df.to_markdown(index=False) for df in clean_tables]
    doc_mmd_tables = [Document(page_content = s, metadata={"source_file" : pdf_dir, "table_num" : i + 1}) for i, s in enumerate(mmd_tables)]
    temp_db = FAISS.from_documents(docs_chunks, medembed_embeddings)
    if tables.n != 0:
        temp_db.add_documents(doc_mmd_tables)
    local_index=FAISS.load_local(folder_path="sampledb_multigen", embeddings=medembed_embeddings, index_name="Multi_Agent_RAG_db", allow_dangerous_deserialization=True)
    local_index.merge_from(temp_db)
    local_index.save_local(folder_path="sampledb_multigen", index_name="Multi_Agent_RAG_db")
    # vectordb_index.save_local("C:\Users\AL56005\Dev\STARS\HPMSdocuments\multi_agent_package\multigen_vecdb", medembed_embeddings)

    return

def the_docs_in_db(vector_index):
    seen = set()
    for doc in vector_index.docstore._dict.values():
        if doc.metadata['source_file'] in seen:
            continue
        else:
            seen.add(doc.metadata['source_file'])
    return seen



def main():
    medembed_embeddings = get_your_embedding()
    vectordb_index = FAISS.load_local(folder_path="sampledb_multigen", embeddings=medembed_embeddings, index_name="Multi_Agent_RAG_db", allow_dangerous_deserialization=True)
    retriever = vectordb_index.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=gpt_4_llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    planner_agent, rag_agent, continuation_agent, user_proxy = collecting_agents(qa_chain)
    groupchat = GroupChat(
        agents = [user_proxy, planner_agent, rag_agent, continuation_agent],
        messages=[],
        max_round=12
    )
    gc_manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    return gc_manager, user_proxy, qa_chain, vectordb_index


    
        