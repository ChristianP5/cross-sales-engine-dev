from langchain_community.document_loaders import PyPDFLoader
from uuid import uuid4

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap

from sentence_transformers.cross_encoder import CrossEncoder

import requests
import os

from langchain_core.documents import Document

def allowed_file(filename, allowedExtensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowedExtensions


def pdf_to_vectorstore(filepath, fileId, vectorStore, purpose):

    loader = PyPDFLoader(filepath)
    
    docs = loader.load()
    ids = [str(uuid4()) for _ in range(len(docs))]

    # For adding metadata to each chunk
    for doc in docs:
        doc.metadata["source_file"] = filepath
        doc.metadata["file_id"] = str(fileId)
        doc.metadata["type"] = "document"
        doc.metadata["purpose"] = purpose


    vectorStore.add_documents(documents=docs, ids=ids)

    return len(docs)

'''
INFERENCE V1
'''

def inference_v1(question, retriever):
    
    RETRIEVED_AMM = 5

    template ="""Role: You are an expert B2B sales strategist and solutions architect specializing in identifying cross-sell opportunities.
Goal: Based on the retrieved context, analyze the customer’s existing technology environment, company best practices, and vendor product portfolio to recommend additional products, services, or upgrades that align with the customer’s business goals and technology stack.
Context Provided: [{context}]
Your Tasks:
1.	Identify potential cross-sell recommendations that complement the customer’s existing environment.
2.	For each recommendation, provide:
o   Product / Service Name
o	Why it fits this customer (alignment with environment, needs, or gaps)
o	Business or technical value (efficiency, performance, ROI, etc.)
o	Level of confidence (High / Medium / Low)
3.	Highlight any dependencies, upgrade paths, or pre-requisites if applicable.
4.	Suggest how to position the recommendation during a sales conversation.
Output Format (Markdown):
### Brief Summary of Customer's Environment:
[1 Paragraph summarizing the Customer's Environment]
### Cross-Sell Recommendations
#### 1. [Product Name]
   * **Fit Rationale:** …
   * **Value Proposition:** …
   * **Confidence Level:** …
   * **Sales Positioning Tip:** …

#### 2. [Product Name]
   * ...
Constraints:
•	Base all insights only on the provided context and retrieved information of Our Products.
•	Do not hallucinate unavailable data; if information is missing, state what additional data would improve accuracy.
•	Use concise, professional, and actionable language suitable for sales enablement documentation.
Question: {question}
"""
 
    prompt = ChatPromptTemplate.from_template(template)

    llm = OllamaLLM(model="llama3.1")

    # Retrieval
    retrieved_docs_raw = retriever.invoke(question)
    
    #  Prepare for Reranking
    retrieved_docs = [] 

    for doc in retrieved_docs_raw:
        retrieved_docs.append(doc.page_content)

    retrieved_docs_ranked, docs_ranked_indices, docs_ranked_scores = rerank(question, retrieved_docs)

    # Get the IDs of the Reranked Documents
    retrieved_docs_ranked_ids = []
    retrieved_docs_ranked_scores = []

    for i in docs_ranked_indices[:RETRIEVED_AMM]:
        
        id = retrieved_docs_raw[i].metadata['file_id']

        retrieved_docs_ranked_ids.append(id)

    for score_raw in docs_ranked_scores[:RETRIEVED_AMM]:
        score = float(score_raw)
        retrieved_docs_ranked_scores.append(score)

    contexts = {
        "ids": retrieved_docs_ranked_ids,
        "scores": retrieved_docs_ranked_scores
    }

    # Augment
    augmented_prompt = prompt.invoke({
        "context": retrieved_docs_ranked[:RETRIEVED_AMM],
        "question": question
    }).to_string()

    generate_chain =  llm | StrOutputParser()
    response = generate_chain.invoke(augmented_prompt)
                                     
    return augmented_prompt, response, contexts


def delete_doc_from_vectorstore(documentId, vectorStore, loggingConfig):
    try:
        vectorStore.delete(
            where={"file_id": str(documentId)}
        )
    except Exception as e:
        loggingConfig["loggingObject"].exception(f"Error when deleting Document {documentId} from Vector Store!")
        print(e)
    
    loggingConfig["loggingObject"].info(f"Document {documentId} deleted Successfully from Vector Store.")
    return True


'''
INFERENCE V2
'''

def inference_v2(question, retriever, loggingConfig, inferenceId, initial_retrieval_prompt=None, llm_memory_formatted=None):
    
    RETRIEVED_AMM_1 = 5
    RETRIEVED_AMM_2 = 5

    '''
    Prompt #1
    '''

    template_1 ="""Role: You are an expert B2B sales strategist and solutions architect specializing in identifying cross-sell opportunities.
Goal: Based on the retrieved context, generate a List of Recommendations based on analyze the customer’s existing technology environment, company best practices, and vendor product portfolio to recommend additional products, services, or upgrades that align with the customer’s business goals and technology stack.
Retrieved Information/Context Provided: {context}
Your Tasks:
1.	Identify Customer's current environnment 
2.  Identify potential Capabilities the Recommended Products should have to complement the customer’s existing environment.
3.	For each recommendation, provide:
o   Capability Description (Up to 1 Paragraph)

Memory (Relevant Past Answers):
{llm_memory}

Output Format (Limit the Ouput to only contain the following):
List of Recommended Capabiities:
- [RECOMMENDATION]
- [RECOMMENDATION]

Example:
List of Recommended Capabilities:
- A Data Warehouse for storing and processing large amounts of data from various Log Sources, Billing Reports, ect.
- A Cloud Native Application Protection Platform to gain visibility and security to Cloud Workloads
- A Managed Database Service for accelerating productivity while minimizing operational overhead.
Constraints:
•	Base all insights only on the provided context and retrieved information.
•	Do not hallucinate unavailable data; if information is missing, state what additional data would improve accuracy.
•	Use concise, professional, and actionable language suitable for sales enablement documentation.
Question: {question}
"""
 
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Pipeline initiated. User Prompt: {question}")
    prompt_1 = ChatPromptTemplate.from_template(template_1)

    llm_1 = OllamaLLM(model="llama3.1")

    # Retrieval 1
    retrieval_question = question
    if initial_retrieval_prompt:
        retrieval_question = initial_retrieval_prompt

    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Retrieval 1 started.")
    retrieved_docs_1_raw = retriever.invoke(retrieval_question)
    retrieved_docs_1 = []
    for docs in retrieved_docs_1_raw:
        retrieved_docs_1.append(docs.page_content)
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Retrieval 1 completed.")

    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Reranking 1 started.")
    context_1, docs_ranked_indices_1, docs_ranked_scores_1 = rerank(question, retrieved_docs_1)

    # Get the IDs of the Reranked Documents
    retrieved_docs_ranked_ids = []
    retrieved_docs_ranked_scores = []

    for score_raw in docs_ranked_scores_1[:RETRIEVED_AMM_1]:
        score = float(score_raw)
        retrieved_docs_ranked_scores.append(score)

    for i in docs_ranked_indices_1[:RETRIEVED_AMM_1]:
        
        id = retrieved_docs_1_raw[i].metadata['file_id']

        retrieved_docs_ranked_ids.append(id)
            

    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Reranking 1 completed.")

    # Augment 1
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Augment 1 started.")
    augmented_prompt_1 = prompt_1.invoke({
        "context": context_1,
        "question": question,
        "llm_memory": llm_memory_formatted
    }).to_string()
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Augment 1 completed.")

    # Generate 1
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Generate 1 started.")
    generate_chain_1 =  llm_1 | StrOutputParser()
    response_1 = generate_chain_1.invoke(augmented_prompt_1)
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Generate 1 completed.")

    '''
    Prompt #2
    '''
    template_2 ="""Role: You are an expert B2B sales strategist and solutions architect specializing in identifying cross-sell opportunities.
    Previously I asked you to recommend Capabilities of Products that can be used to improve the Customer's Environment.
    Which you previously answered:
    {response_1}
    
    
    Goal: Explain why did you Recommend me the given Capabilities and from the given list of our Products, which can provide those capabilities?
    
    Retrieved Information/Context about our Customer: [{context_1}]
    Retrieved Information/Context about our Products: [{context_2}]
    Your Tasks:
    1.	Answer the Question: {question}
    2.  Identify potential cross-sell recommendations that complement the customer’s existing environment.
    3.	For each recommendation, provide:
    o   Product / Service Name
    o	Why it fits this customer (alignment with environment, needs, or gaps)
    o	Business or technical value (efficiency, performance, ROI, etc.)
    o	Level of confidence (High / Medium / Low)
    3.	Highlight any dependencies, upgrade paths, or pre-requisites if applicable.
    4.	Suggest how to position the recommendation during a sales conversation.
    
    Memory (Relevant Past Answers):
    {llm_memory}
    
    Output Format (Markdown):
    ### Brief Summary of Customer's Environment:
    [1 Paragraph summarizing the Customer's Environment]
    ### Cross-Sell Recommendations
    #### 1. [Product Name]
    * **Short Description:** …
    * **Fit Rationale:** …
    * **Value Proposition:** …
    * **Confidence Level:** …
    * **Sales Positioning Tip:** …
    
    #### 2. [Product Name]
    * ...
    Constraints:
    •	Base all insights only on the provided context and retrieved information.
    •	Do not hallucinate unavailable data; if information is missing, state what additional data would improve accuracy.
    •	Use concise, professional, and actionable language suitable for sales enablement documentation.
    •	Give a relevant Contact Person according to PT Multipolar Tehcnology's Guidelines that can help answer the question better.
    
    """
    prompt_2 = ChatPromptTemplate.from_template(template_2)

    llm_2 = OllamaLLM(model="llama3.1")

    # Retrieve 2
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Retrieve 2 started.")
    retrieved_docs_2_raw = retriever.invoke(response_1)
    retrieved_docs_2 = []
    for docs in retrieved_docs_2_raw:
        retrieved_docs_2.append(docs.page_content)
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Retrieve 2 completed.")

    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Reranking 2 started.")
    context_2, docs_ranked_indices_2, docs_ranked_scores_2 = rerank(response_1, retrieved_docs_2)

    for score_raw in docs_ranked_scores_2[:RETRIEVED_AMM_2]:
        score = float(score_raw)
        retrieved_docs_ranked_scores.append(score)

    for i in docs_ranked_indices_2[:RETRIEVED_AMM_2]:
        
        id = retrieved_docs_2_raw[i].metadata['file_id']
        retrieved_docs_ranked_ids.append(id)

    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Reranking 2 completed.")

    # Augment 2
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Augment 2 started.")
    augmented_prompt_2 = prompt_2.invoke({
        "response_1": response_1,
        "context_1": context_1,
        "context_2": context_2,
        "question": question,
        "llm_memory": llm_memory_formatted
        }).to_string()
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Augment 2 completed.")

    # Generate 2
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Generate 2 started.")
    generate_chain_2 =  llm_2 | StrOutputParser()
    response_2 = generate_chain_2.invoke(augmented_prompt_2)
    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Generate 2 completed.")

    loggingConfig["loggingObject"].info(f"[V2 | Inference: {inferenceId}] Pipeline completed.")
    
    contexts = {
        "ids": retrieved_docs_ranked_ids,
        "scores": retrieved_docs_ranked_scores
    }
    
    return response_2, augmented_prompt_1, augmented_prompt_2, contexts


'''
INFERENCE V3
'''

def inference_v3(question, retriever, loggingConfig, inferenceId, llm_memory_formatted=None, chatId=None, initial_retrieval_prompt=None):
    
    RETRIEVED_AMM = 5

    template ="""Role: You are an expert B2B sales strategist and solutions architect specializing in identifying cross-sell opportunities.
Goal: Answer the Question according to the provided Context
Context Provided: [{context}]
Memory (Relevant Past Answers):
{llm_memory}

Example Output Format (Markdown):
### Products inside Company ABC
#### 1. [Product Name]
   * **Description:** …
   * **Other Information:** …

#### 2. [Product Name]
   * ...
Constraints:
•	Base all insights only on the provided context and retrieved information.
•	Do not hallucinate unavailable data; if the provided context doesn't contain the relevant information to answer the question, state what additional data would improve accuracy.
•	Use concise, professional, and actionable language suitable for sales enablement documentation.
•	Give a relevant Contact Person according to PT Multipolar Tehcnology's Guidelines that can help answer the question better.
Question: {question}
"""
 
    prompt = ChatPromptTemplate.from_template(template)

    # Retrieval
    retrieval_question = question
    if initial_retrieval_prompt:
        retrieval_question = initial_retrieval_prompt

    loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Augment 1 start.")
    retrieved_docs_raw = retriever.invoke(retrieval_question)
    loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Augment 1 finished.")
    
    #  Prepare for Reranking
    loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Reranking 1 start.")
    retrieved_docs = [] 

    for doc in retrieved_docs_raw:
        retrieved_docs.append(doc.page_content)

    retrieved_docs_ranked, docs_ranked_indices, docs_ranked_scores = rerank(question, retrieved_docs)

    # Get the IDs of the Reranked Documents
    retrieved_docs_ranked_ids = []
    retrieved_docs_ranked_scores = []

    for i in docs_ranked_indices[:RETRIEVED_AMM]:
        
        id = retrieved_docs_raw[i].metadata['file_id']

        retrieved_docs_ranked_ids.append(id)

    for score_raw in docs_ranked_scores[:RETRIEVED_AMM]:
        score = float(score_raw)
        retrieved_docs_ranked_scores.append(score)

    contexts = {
        "ids": retrieved_docs_ranked_ids,
        "scores": retrieved_docs_ranked_scores
    }

    loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Reranking 1 finished.")

    # Augment
    loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Augment 1 start.")
    augmented_prompt = prompt.invoke({
        "context": retrieved_docs_ranked[:RETRIEVED_AMM],
        "question": question,
        "llm_memory": llm_memory_formatted
    }).to_string()
    loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Augment 1 finished.")

    # Generate
    loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Generate 1 start.")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT = "https://c-ailab-aifoundry1.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"{augmented_prompt}"
            }
        ],
        "max_completion_tokens": 40000,
        "model": "o4-mini"
    }

    try:
        response_raw = requests.post(AZURE_ENDPOINT, headers=headers, json=payload)
        response = response_raw.json()["choices"][0]["message"]["content"]
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Something went wrong when inferencing through the Azure AI Model Inference API.")
        print(e)

    loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Generate 1 finished.")

    loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Pipeline completed.")
    
                                     
    return response, augmented_prompt, contexts

def rerank(question, docs):
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
 
    ranks = model.rank(question, docs)
    docs_ranked = []
    docs_ranked_indices = []
    docs_ranked_scores = []

    for rank in ranks:
        docs_ranked.append(docs[rank["corpus_id"]])
        docs_ranked_indices.append(rank["corpus_id"])
        docs_ranked_scores.append(rank["score"])

    return docs_ranked, docs_ranked_indices, docs_ranked_scores

'''
CHAT V1

'''
def chat_v1(prompt, retriever, loggingConfig, inferenceId, vector_store, chatId):

    # 1) Classify Prompt
    # 2) Pre-Process Prompt
    # 3) Get the Chat's Memory
    # 4) Generate Response

    # 1)
    classifications = {
        "cross-sell": """
Choose this ONLY if the user is explicitly asking for:
- product recommendations
- product suggestions
- items to buy
- what to offer a customer
- what products go well with something
- what to use in a specific customer scenario""",
        "others": """
Choose this if the user is asking about:
- definitions of cross-selling
- how to perform cross-selling
- examples of cross-selling
- training, explanations, or general knowledge
- anything not requesting specific product recommendations
- information about customers
"""
    }
    response = questionClassifier(prompt, classifications, inferenceId, loggingConfig)
    
    
    # 2)
    initial_retrieval_prompt = f"{prompt}. Make sure to follow guidelines provided by PT Multipolar Technology."

    # 3)
    # Get the LLM Memory for the Chat
    if vector_store:
        loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Retrieving LLM Memory for Chat {chatId}.")
        # Build the Retriever
        llm_memory_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "filter": {
                    "$and": [
                        {"type": "llm_memory"},
                        {"chatId": str(chatId)}
                    ]
                }
            }
        )

        try:
            retrieved_llm_memory_raw = llm_memory_retriever.invoke(prompt)

            retrieved_llm_memory = []
            for item in retrieved_llm_memory_raw:
                retrieved_llm_memory.append(item.page_content)
            
            loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Successfully Retrieved LLM Memory for Chat {chatId}.")
            
        except Exception as e:
            print(e)
            loggingConfig["loggingObject"].info(f"[V3 | Inference: {inferenceId}] Failed Retrieved LLM Memory for Chat {chatId}.")
        
        
        # print(f"retrieved_llm_memory: {str(retrieved_llm_memory)}")
        llm_memory_formatted = "\n".join(
            [
                f"[#{i+1}{' - Most Relevant' if i==0 else ''}]:\n{memory_item}"
                for i, memory_item in enumerate(retrieved_llm_memory)
            ]
        )

        # print(f"llm_memory_formatted: {llm_memory_formatted}")
        

    # 4)
    if "cross-sell" in response:
        loggingConfig["loggingObject"].info(f"[Chat V1 | Chat: {chatId} Inference: {inferenceId}] Inference is classified as CROSS-SELL")
        response_raw, augmented_prompt_1, augmented_prompt_2, contexts = inference_v2(prompt, retriever, loggingConfig, inferenceId, initial_retrieval_prompt, llm_memory_formatted)
        augmented_prompt = augmented_prompt_2

    else:
        loggingConfig["loggingObject"].info(f"[Chat V1 | Chat: {chatId} Inference: {inferenceId}] Inference is classified as OTHERS")
        response_raw, augmented_prompt, contexts = inference_v3(prompt, retriever, loggingConfig, inferenceId, llm_memory_formatted, chatId, initial_retrieval_prompt)

    return response_raw, augmented_prompt, contexts


'''
Question Classifier feature
'''
def questionClassifier(question, classifications, inferenceId, loggingConfig, relevant_llm_memory="" ,recent_llm_memory=""):
    
    classifications_list = "\n".join([f"- {key}: {value}" for key,value in classifications.items()])
    template ="""
Task:
You are a strict Question Classifier.

Objective:
Classify the user's Question into exactly ONE category by returning ONLY the key from the provided list.

Allowed Output:
• Return ONLY one classification key
• No explanations, no punctuation, no formatting

Classification Keys:
{classifications}

---

Context Rules:
• You MAY use conversation history ONLY to:
  - Resolve pronouns ("it", "they", "this", etc.)
  - Understand implied intent or subject
• Do NOT answer the question
• Do NOT introduce new information
• Do NOT infer intent beyond what is stated or clearly implied

---

Recent Conversation (Most recent messages):
{recent_conversation}

Relevant Past Conversation (Semantically related):
{relevant_conversation}

---

Examples:

Example Question #1:
What is GCP?
Response:
others

Example Question #2:
What can I recommend to Company ABC?
Response:
cross-sell

---

Question:
{question}

Answer:
"""

    prompt = ChatPromptTemplate.from_template(template)

    final_prompt = prompt.invoke({
        'question': question,
        'classifications': classifications_list,
        'recent_conversation': recent_llm_memory,
        'relevant_conversation': relevant_llm_memory
        }).to_string()
    
    print(f"final_prompt: {final_prompt}")

    
    loggingConfig["loggingObject"].info(f"[Chat V1 | Inference: {inferenceId}] Question Classification start.")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT = "https://c-ailab-aifoundry1.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"{final_prompt}"
            }
        ],
        "max_completion_tokens": 40000,
        "model": "o4-mini"
    }

    try:
        response_raw = requests.post(AZURE_ENDPOINT, headers=headers, json=payload)
        # print(response_raw.json())
        response = response_raw.json()["choices"][0]["message"]["content"]
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat V1 | Inference: {inferenceId}] Something went wrong when performing Questin Classification through the Azure AI Model Inference API.")
        print(e)

    loggingConfig["loggingObject"].info(f"[Chat V1 | Inference: {inferenceId}] Question Classification completed.")
    return response

'''
for Memory Feature
'''
def memory_to_vectorstore(inferenceId, question, response, chatId, vectorStore, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Chat V1 | Chat: {chatId} Inference: {inferenceId}] Adding Inference to Memory of Chat {chatId}.")
    
    page_content = f"Question:\n{question}\nAnswer:\n{str(response)}"
    
    doc = Document(
        page_content=page_content,
    )

    doc.metadata["type"] = "llm_memory"
    doc.metadata["purpose"] = "llm_memory"
    doc.metadata["inferenceId"] = str(inferenceId)
    doc.metadata["chatId"] = str(chatId)
    doc.metadata["question"] = str(question)

    print(f"Adding to Memory. Question: {str(question)}")

    try:
        memory_id = str(uuid4())
        vectorStore.add_documents(documents=[doc], ids=[memory_id])

    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat V1 | Chat: {chatId} Inference: {inferenceId}] Failed adding Inference to Memory of Chat {chatId}.")
        print(e)
        return True
    
    loggingConfig["loggingObject"].info(f"[Chat V1 | Chat: {chatId} Inference: {inferenceId}] Inference Added to Memory of Chat {chatId}.")

    return True

'''
Customer Management Utils
'''
def generateCustomerProfile(customerId, vectorStore, loggingConfig, instructionConfig):
    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile start.")

    # Collect Parameters for the Prompt
    role = instructionConfig["role"]
    objective = instructionConfig["objective"]
    examples = instructionConfig["examples"]
    outputFormat = instructionConfig["outputFormat"]

    # Retrieval
    # Build the Retriever
    retriever = vectorStore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10,
            "filter": {
                "$and": [
                    {"type": "document"},
                    {"purpose": str(customerId)}
                ]
            }
        }
    )
    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Retrieval Start.")

    try:
        retrieved_context_raw = retriever.invoke(objective)

        retrieved_context = []
        for item in retrieved_context_raw:
            retrieved_context.append(item.page_content)
            
        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Retrieval Success.")
        # print(retrieved_context)

    except Exception as e:
        print(e)
        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Retrieval Error.")
        raise Exception(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Retrieval Error.")

    # Build the Prompt
    template = """
    Role:
    {role}

    Objective:
    {objective}

    Expected Output Format:
    {outputFormat}

    Examples:
    {examples}

    Context:
    {context}
    """
    
    # Augement
    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Augment Start.")
    prompt = ChatPromptTemplate.from_template(template)

    final_prompt = prompt.invoke({
        'role': role,
        'objective': objective,
        'outputFormat': outputFormat,
        'examples': examples,
        'context': retrieved_context
        }).to_string()
    
    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Augment Finished.")

    # Generate
    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Generate Start.")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT = "https://c-ailab-aifoundry1.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"{final_prompt}"
            }
        ],
        "max_completion_tokens": 100000,
        "model": "o4-mini"
    }

    try:
        response_raw = requests.post(AZURE_ENDPOINT, headers=headers, json=payload)
        # print(response_raw.json())
        response = response_raw.json()["choices"][0]["message"]["content"]
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Generate Error.")
        print(e)
        raise Exception(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Generate Error.")


    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Generate Customer Profile - Generate Success.")
    
    return response


'''
CHAT V2
'''

def chat_v2(prompt, retriever, loggingConfig, inferenceId, vector_store, chatId, recent_llm_memory_input):

    RETRIEVED_AMM = 5

    # 1) Classify Prompt
    # 2) Pre-Process Prompt
    # 3) Get the Recent Memory for the Chat
    # 4) Get the Relevant Memory for the Chat
    # 5) Get the Relevant Regulation Documents for the Question
    # 6) Generate Response

    # 1)
    classifications = {
        "cross-sell": """
Choose this ONLY if the user is explicitly asking for:
- product recommendations
- product suggestions
- items to buy
- what to offer a customer
- what products go well with something
- what to use in a specific customer scenario""",
        "others": """
Choose this if the user is asking about:
- definitions of cross-selling
- how to perform cross-selling
- examples of cross-selling
- training, explanations, or general knowledge
- anything not requesting specific product recommendations
- information about customers
"""
    }
    
    
    
    # 2)
    initial_retrieval_prompt = f"{prompt}. Make sure to follow guidelines provided by PT Multipolar Technology."

    # 3)
    # Get the Recent LLM Memory for the Chat
    recent_llm_memory = recent_llm_memory_input 

    # 4)
    # Get the Relevant LLM Memory for the Chat
    if vector_store:
        loggingConfig["loggingObject"].info(f"[Chat V2 | Inference: {inferenceId}] Retrieving LLM Memory for Chat {chatId}.")
        # Build the Retriever
        llm_memory_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "filter": {
                    "$and": [
                        {"type": "llm_memory"},
                        {"chatId": str(chatId)}
                    ]
                }
            }
        )

        try:
            retrieved_llm_memory_raw = llm_memory_retriever.invoke(prompt)

            retrieved_llm_memory = []
            for item in retrieved_llm_memory_raw:
                retrieved_llm_memory.append(item.page_content)
            
            loggingConfig["loggingObject"].info(f"[Chat V2 | Inference: {inferenceId}] Successfully Retrieved LLM Memory for Chat {chatId}.")
            
        except Exception as e:
            print(e)
            loggingConfig["loggingObject"].info(f"[Chat V2 | Inference: {inferenceId}] Failed Retrieved LLM Memory for Chat {chatId}.")
        
        
        # print(f"retrieved_llm_memory: {str(retrieved_llm_memory)}")
        llm_memory_formatted = "\n".join(
            [
                f"[#{i+1}{' - Most Relevant' if i==0 else ''}]:\n{memory_item}"
                for i, memory_item in enumerate(retrieved_llm_memory)
            ]
        )

        # print(f"llm_memory_formatted: {llm_memory_formatted}")
        

    # 5)
    contexts = {
        "ids": [],
        "scores": []
    }

    if vector_store:
        loggingConfig["loggingObject"].info(f"[Chat V2 | Inference: {inferenceId}] Retrieving Regulation Documents for Chat {chatId}.")
        # Build the Retriever
        regulationDocs_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "filter": {
                    "$and": [
                        {"type": "document"},
                        {"purpose": "REGULATION"}
                    ]
                }
            }
        )

        try:
            retrievedRegulationDocs_raw = regulationDocs_retriever.invoke(initial_retrieval_prompt)

            retrievedRegulationDocs = []
            for item in retrievedRegulationDocs_raw:
                retrievedRegulationDocs.append(item.page_content)
            
            loggingConfig["loggingObject"].info(f"[Chat V2 | Inference: {inferenceId}] Successfully Retrieved Regulation Documents for Chat {chatId}.")
            
        except Exception as e:
            print(e)
            loggingConfig["loggingObject"].info(f"[Chat V2 | Inference: {inferenceId}] Failed Retrieved Regulation Documents for Chat {chatId}.")
        
        
        #  Prepare for Reranking
        loggingConfig["loggingObject"].info(f"[Chat V2 | Inference: {inferenceId}] Reranking 1 start.")

        # Rerank
        retrievedRegulationDocs_ranked, regulationDocs_ranked_indices, regulationnDocs_ranked_scores = rerank(prompt, retrievedRegulationDocs)
        
        # Get the IDs of the Reranked Documents
        retrieved_regulationDocs_ranked_ids = []
        retrieved_regulationDocs_ranked_scores = []

        for i in regulationDocs_ranked_indices[:RETRIEVED_AMM]:
            
            id = retrievedRegulationDocs_raw[i].metadata['file_id']

            retrieved_regulationDocs_ranked_ids.append(id)

        for score_raw in regulationnDocs_ranked_scores[:RETRIEVED_AMM]:
            score = float(score_raw)
            retrieved_regulationDocs_ranked_scores.append(score)

        # Add the Retrieved Context to the Contexts List
        for ids in retrieved_regulationDocs_ranked_ids:
            contexts["ids"].append(ids)
        
        for scores in retrieved_regulationDocs_ranked_scores:
            contexts["scores"].append(scores)

        # print(f"retrieved_llm_memory: {str(retrieved_llm_memory)}")
        regulationDocs_formatted = "\n".join(
            [
                f"Regulations Document #{i+1}:\n{regulation_item}"
                for i, regulation_item in enumerate(retrievedRegulationDocs_ranked)
            ]
        )



    # 6)
    # Question Classifier
    response = questionClassifier(prompt, classifications, inferenceId, loggingConfig, llm_memory_formatted, recent_llm_memory)

    if "cross-sell" in response:
        loggingConfig["loggingObject"].info(f"[Chat V2 | Chat: {chatId} Inference: {inferenceId}] Inference is classified as CROSS-SELL")
        response_raw, augmented_prompt_1, augmented_prompt_2, contexts = crossSellPipeline_chat_v2(prompt, retriever, loggingConfig, inferenceId, initial_retrieval_prompt, llm_memory_formatted, regulationDocs_formatted)
        augmented_prompt = augmented_prompt_2
        
    else:
        loggingConfig["loggingObject"].info(f"[Chat V2 | Chat: {chatId} Inference: {inferenceId}] Inference is classified as OTHERS")
        response_raw, augmented_prompt, contexts = defaultPipeline_chat_v2(prompt, retriever, loggingConfig, inferenceId, llm_memory_formatted, chatId, initial_retrieval_prompt, recent_llm_memory, regulationDocs_formatted, contexts)

    return response_raw, augmented_prompt, contexts

def crossSellPipeline_chat_v2():
    return 0

def defaultPipeline_chat_v2(question, retriever, loggingConfig, inferenceId, llm_memory_formatted, chatId, initial_retrieval_prompt, recent_llm_memory, regulationDocs, contexts):
    
    RETRIEVED_AMM = 5

    template ="""Role:
You are an expert B2B sales strategist and solutions architect operating under strict company regulations.

Primary Objective:
Answer the user's question accurately while STRICTLY complying with Company Regulation Documents.

MANDATORY SOURCE PRIORITY (DO NOT VIOLATE):
1. Company Regulation Documents (highest authority)
2. Retrieved Context (RAG results)
3. Direct Previous Conversation
4. Your Relevant Past Answers
5. General knowledge (ONLY if not restricted and clearly allowed)

If any lower-priority source conflicts with a higher-priority one:
• You MUST follow the higher-priority source
• You MUST explicitly state the restriction if it affects the answer

---

PRONOUN RESOLUTION RULE (MANDATORY):
• When resolving pronouns ("it", "they", "this"):
  - ALWAYS prefer the most recent explicitly mentioned primary entity
  - Ignore older or repeated entities unless the user explicitly references them
• Do NOT resolve pronouns based on frequency or relevance alone
• If multiple entities are equally recent:
  - Ask for clarification OR state ambiguity

---

Company Regulation Documents:
{regulation_context}

Retrieved Business / Customer Context:
{context}

Direct Previous Conversation (STRICT TURN ORDER — highest priority):
{recent_llm_memory}

Relevant Past Answers (REFERENCE ONLY — DO NOT USE FOR PRONOUN RESOLUTION):
{relevant_llm_memory}

---

Response Rules:
• Do NOT disclose restricted, confidential, or non-approved information
• Do NOT infer or guess regulated information
• If the question violates policy, respond with a compliant explanation
• Use professional, sales-appropriate language
• Provide PT Multipolar Technology contact persons ONLY if permitted by regulation

---

Output Format (Markdown):

### Answer
[Direct answer to the question]

### Reasoning (Brief)
• Identified subject: ...
• Supporting context: ...

### Data Gaps (if any)
• ...

---

Question:
{question}
"""
 
    prompt = ChatPromptTemplate.from_template(template)

    # Retrieval
    retrieval_question = question
    if initial_retrieval_prompt:
        retrieval_question = initial_retrieval_prompt

    loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Augment 1 start.")
    retrieved_docs_raw = retriever.invoke(retrieval_question)
    loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Augment 1 finished.")
    
    #  Prepare for Reranking
    loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Reranking 1 start.")
    retrieved_docs = [] 

    for doc in retrieved_docs_raw:
        retrieved_docs.append(doc.page_content)

    # Rerank
    retrieved_docs_ranked, docs_ranked_indices, docs_ranked_scores = rerank(question, retrieved_docs)

    # Get the IDs of the Reranked Documents
    retrieved_docs_ranked_ids = []
    retrieved_docs_ranked_scores = []

    for i in docs_ranked_indices[:RETRIEVED_AMM]:
        
        id = retrieved_docs_raw[i].metadata['file_id']

        retrieved_docs_ranked_ids.append(id)

    for score_raw in docs_ranked_scores[:RETRIEVED_AMM]:
        score = float(score_raw)
        retrieved_docs_ranked_scores.append(score)

    # Add the Retrieved Context to the Contexts List
    for ids in retrieved_docs_ranked_ids:
        contexts["ids"].append(ids)
    
    for scores in retrieved_docs_ranked_scores:
        contexts["scores"].append(scores)

    loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Reranking 1 finished.")

    # Augment
    loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Augment 1 start.")
    augmented_prompt = prompt.invoke({
        "context": retrieved_docs_ranked[:RETRIEVED_AMM],
        "question": question,
        "relevant_llm_memory": llm_memory_formatted,
        "recent_llm_memory": recent_llm_memory,
        "regulation_context": regulationDocs
    }).to_string()
    loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Augment 1 finished.")

    # Generate
    loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Generate 1 start.")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT = "https://c-ailab-aifoundry1.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"{augmented_prompt}"
            }
        ],
        "max_completion_tokens": 40000,
        "model": "o4-mini"
    }

    try:
        response_raw = requests.post(AZURE_ENDPOINT, headers=headers, json=payload)
        response = response_raw.json()["choices"][0]["message"]["content"]
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Something went wrong when inferencing through the Azure AI Model Inference API.")
        print(e)

    loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Generate 1 finished.")

    loggingConfig["loggingObject"].info(f"[Chat V2 - others | Inference: {inferenceId}] Pipeline completed.")
    
                                     
    return response, augmented_prompt, contexts