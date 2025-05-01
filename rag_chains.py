from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_summary_chain():
    summarize_template = """ 
        You are an assistant tasked with summarizing tables and text. \ 
        Give a concise summary of the table or text. Table or text chunk: {element}
    """
    summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125-openai",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key='LLMah53026ty*7%!p0GLD',  
        base_url='https://llm-uat.elevancehealth.com/api/openai'
    )
    summarize_chain = {"element" : lambda x: x} | summarize_prompt | llm | StrOutputParser()
    return summarize_chain

def get_contextualize_chain(llm):
    contextualize_system_prompt = (
        """Given a chat history and the latest user question 
        which might reference context in the chat history, 
        formulate a standalone question which can be understood 
        without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""
    )
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        ("user", "{chat_history}"),
        ("human", "{input}")
    ])
    contextualize_chain = (
        contextualize_prompt
        | llm
        | (lambda msg:msg.content)
    )
    return contextualize_chain

# @chain
# def history_aware_qa(input, retriever):
#     if input.get('chat_history'):
#         question = contextualize_chain.invoke(input)
#     else:
#         question = input['input']
#     context = retriever.invoke(question)

#     return qa_chain.invoke({
#         **input,
#         "context" : context
#     })

# def get_grader_chain_w_history(retriever, llm, parser):
#     grader_template = """
#         Grade the following retrieved chunks which can include text, tables as well as the chat history from the user : 
#         {context}
#         according to whether it answers the following user query satisfactorily:
#         {question}
#         Chat History : {chat_history}
#         Return the answer as a JSON object using the key relevant and item value yes or no.
#     """
    
#     grader_prompt = ChatPromptTemplate.from_template(grader_template)
    
#     grader_chain = (
#         {"context" : retriever, "question" : RunnablePassthrough()}
#         | {"chat_history" : RunnablePassthrough()}
#         | grader_prompt
#         | llm
#         | JsonOutputParser()
#     )
#     return grader_chain

def get_grader_chain(retriever, llm, parser):
    grader_template = """
        Grade the following retrieved chunks which can include text and tables : 
        {context}
        according to whether it answers the following user query satisfactorily:
        {question}
        Return the answer as a JSON object using the key relevant and item value yes or no.
    """
    
    grader_prompt = ChatPromptTemplate.from_template(grader_template)
    
    grader_chain = (
        {"context" : retriever, "question" : RunnablePassthrough()}
        | grader_prompt
        | llm
        | JsonOutputParser()
    )
    return grader_chain

# def get_gen_chain_w_history(retriever, llm, parser):
#     gen_template = """
#         You are a document generator that utilizes the context below to answer the user's question in three different ways. 
#                   The main aim for this exercise is to generate three varied answers to the user's question. Ensure answers:
#                   - Include mathematical formulas written in proper mathematical notation wherever applicable, formatted for clarity.
#                   - Provide as much detailed responses as possible based solely on the retrieved context.
#                   - Use subtopics and bullet points where needed for better organization.
#                   - Add additional information related to the question based on the retrieved chunk.
#                   - Use the metadata from the retrieved chunks to append to the answer the page number and the source file of the answer provided
#                   Question: {question}
#                   Context: {context}
#                   Chat History : {chat_history}
#                   Return a JSON object containing the three answers, following the example below:
                  
#                       {{
#                           "1": "Answer 1 content",
#                           "2": "Answer 2 content",
#                           "3": "Answer 3 content"
#                       }}
#     """
#     gen_prompt = ChatPromptTemplate.from_template(gen_template)
#     gen_chain = (
#         {"context" : retriever, "question" : RunnablePassthrough(), "chat_history" : RunnablePassthrough()}
#         | gen_prompt
#         | llm
#         | parser
#     )
#     return gen_chain

def get_gen_chain(retriever, llm, parser):
    """
        This function creates a generator chain to generate three different answers based on the retrieved context, the next step involves rating these generated answers to select the best answers,
        Parameters : the (multivector) retriever, llm model (your choice), parser (prefer JSON)
        Returns : generator chain which can be used (.invoke() method)
                {"1" : <answer>, "2" : <answer>, "3" : <answer>}
    """

    gen_template = """
        You are a document generator that utilizes the context below to answer the user's question in three different ways. 
                  The main aim for this exercise is to generate three varied answers to the user's question. Ensure answers:
                  - Include mathematical formulas written in proper mathematical notation wherever applicable, formatted for clarity.
                  - Provide as much detailed responses as possible based solely on the retrieved context.
                  - Use subtopics and bullet points where needed for better organization.
                  - Add additional information related to the question based on the retrieved chunk.
                  - Use the metadata from the retrieved chunks to append to the answer the page number and the source file of the answer provided
                  Question: {question}
                  Context: {context}
                  Return a JSON object containing the three answers, following the example below:
                  
                      {{
                          "1": "Answer 1 content",
                          "2": "Answer 2 content",
                          "3": "Answer 3 content"
                      }}
    """
    gen_prompt = ChatPromptTemplate.from_template(gen_template)
    gen_chain = (
        {"context" : retriever, "question" : RunnablePassthrough()}
        | gen_prompt
        | llm
        | parser
    )
    return gen_chain

def ai_eval_chain(llm, parser):
    """
        This function creates an ai evaluator chain that judges the generated answers based on the ratings given below (1-5), only chain which uses PromptTemplate instead of ChatPromptTemplate,
        Parameters : llm, and parser (JSON)
        Returns the eval chain (.invoke method)
    """
    eval_prompt = PromptTemplate(
        template = """
                    You are an AI evaluator responsible for grading responses to a user query.
                    Evaluation Criteria:
                    1 : Completely irrelevant: No relevant content related to the query.
                    2 : Weakly relevant: Somewhat related but lacks context or coherence.
                    3 : Minimally Satisfactory: Basic answer but lacks direct supporting evidence.
                    4 : Strong Answer: Clear, mostly accurate, with partial supporting evidence.
                    5 : Highly relevant: Fully answers the question with strong supporting details.

                    You will receive a query and three answers. Evaluate each independently, provide a score, a short justification, and select the best response.

                    Question: {question}

                    Response 1: {response_1}
                    Response 2: {response_2}
                    Response 3: {response_3}

                    Return a JSON object {{ 
                        "scores": {{
                            "1": {{"score": <score>, "justification": "<justification>"}},
                            "2": {{"score": <score>, "justification": "<justification>"}},
                            "3": {{"score": <score>, "justification": "<justification>"}}
                        }},
                        "best_answer": "<best_answer>"
                    }}
                    """,
        input_variables=["question", "response_1", "response_2", "response_3"],
        partial_variables={"format_inst" : parser.get_format_instructions()},
    )
    eval_chain = eval_prompt | llm | parser
    return eval_chain

# def ai_eval_chain_w_history(llm, parser):
#     eval_prompt = PromptTemplate(
#         template = """
#                     You are an AI evaluator responsible for grading responses to a user query.
#                     Evaluation Criteria:
#                     1 : Completely irrelevant: No relevant content related to the query.
#                     2 : Weakly relevant: Somewhat related but lacks context or coherence.
#                     3 : Minimally Satisfactory: Basic answer but lacks direct supporting evidence.
#                     4 : Strong Answer: Clear, mostly accurate, with partial supporting evidence.
#                     5 : Highly relevant: Fully answers the question with strong supporting details.

#                     You will receive a query and three answers. Evaluate each independently, provide a score, a short justification, and select the best response.

#                     Question: {question}
#                     Chat History : {chat_history}
#                     Response 1: {response_1}
#                     Response 2: {response_2}
#                     Response 3: {response_3}

#                     Return a JSON object {{ 
#                         "scores": {{
#                             "1": {{"score": <score>, "justification": "<justification>"}},
#                             "2": {{"score": <score>, "justification": "<justification>"}},
#                             "3": {{"score": <score>, "justification": "<justification>"}}
#                         }},
#                         "best_answer": "<best_answer>"
#                     }}
#                     """,
#         input_variables=["question", "chat_history", "response_1", "response_2", "response_3"],
#         partial_variables={"format_inst" : parser.get_format_instructions()},
#     )
#     eval_chain = eval_prompt | llm | parser
#     return eval_chain

# def wrapper_chain_w_history(query, gpt_3_5_llm, gpt_4_llm, retriever, chat_history):
#     contextualize_chain = get_contextualize_chain(gpt_3_5_llm)

#     @chain
#     def history_aware_qa(input):
#         if input.get('chat_history'):
#             question = contextualize_chain.invoke(input)
#         else:
#             question = input['input']
        
#         context = retriever.invoke(question)
#         gen_chain = get_gen_chain_w_history(retriever, gpt_4_llm, JsonOutputParser())
#         gen_responses=gen_chain.invoke({
#             "question":query, "chat_history" : chat_history
#         })
#         eval_chain = ai_eval_chain_w_history(gpt_4_llm, JsonOutputParser())
#         eval_response = eval_chain.invoke({
#             "question" : query,
#             "chat_history" : chat_history,
#             "response_1" : gen_responses["1"],
#             "response_2" : gen_responses["2"],
#             "response_3" : gen_responses["3"],
#         })
#         return gen_responses[f"{eval_response['best_answer']}"]
#     qa_with_history = RunnableWithMessageHistory(
#         history_aware_qa, 
#         lambda _ : InMemoryChatMessageHistory(),
#         inpurt_message_key="input",
#         history_message_key="chat_history"
#     )
#     final_response = qa_with_history.invoke(
#         {"input" : query},
#         config={"configurable" : {"session_id" : "123"}}
#     )
#     return final_response
#     # if chat_history:
#     #     query=contextualize_chain.invoke({"chat_history" : chat_history, "input" : query})

#     # grader_chain = get_grader_chain(retriever, gpt_3_5_llm, JsonOutputParser())
#     # grader_response = grader_chain.invoke({"question":query, "chat_history" : chat_history})
#     # if grader_response['relevant'] == 'no':
#     #     return "The document grader has judged that the context is not enough to answer the question, hence UNANSWERABLE"
    
#     # gen_chain = get_gen_chain(retriever, gpt_4_llm, JsonOutputParser())
#     # gen_responses = gen_chain.vinoke({"question" : query, "chat_history" : chat_history})


#     # eval_chain = ai_eval_chain(gpt_4_llm, JsonOutputParser())
#     # eval_response = eval_chain.invoke({
#     #     "question" : query,
#     #     "chat_history" : chat_history,
#     #     "response_1" : gen_responses["1"],
#     #     "response_2" : gen_responses["2"],
#     #     "response_3" : gen_responses["3"],
#     # })
#     # return gen_responses[f"{eval_response['best_answer']}"]

def wrapper_chain(query, gpt_3_5_llm, gpt_4_llm, retriever):
    grader_chain = get_grader_chain(retriever, gpt_3_5_llm, JsonOutputParser())
    grader_response = grader_chain.invoke(query)
    if grader_response['relevant'] == 'no':
        return "The document grader has judged that the context is not enough to answer the question, hence UNANSWERABLE"
    gen_chain = get_gen_chain(retriever, gpt_4_llm, JsonOutputParser())
    gen_responses = gen_chain.invoke(query)
    eval_chain = ai_eval_chain(gpt_4_llm, JsonOutputParser())
    eval_response = eval_chain.invoke({"question" : query, 
                                       "response_1" : gen_responses["1"], 
                                       "response_2" : gen_responses["2"], 
                                       "response_3" : gen_responses["3"]}) 
    return gen_responses[f"{eval_response['best_answer']}"]

