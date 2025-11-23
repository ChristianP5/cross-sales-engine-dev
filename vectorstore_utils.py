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

def allowed_file(filename, allowedExtensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowedExtensions


def pdf_to_vectorstore(filepath, fileId, vectorStore):

    loader = PyPDFLoader(filepath)
    
    docs = loader.load()
    ids = [str(uuid4()) for _ in range(len(docs))]

    # For adding metadata to each chunk
    for doc in docs:
        doc.metadata["source_file"] = filepath
        doc.metadata["file_id"] = str(fileId)


    vectorStore.add_documents(documents=docs, ids=ids)

    return len(docs)

'''
INFERENCE V1
'''

def inference_v1(question, retriever):
    
    RETRIEVED_AMM = 3

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
    ids = []
    try:
        ids.append(documentId)
        vectorStore.delete(ids=ids)
    except Exception as e:
        loggingConfig["loggingObject"].exception(f"Error when deleting Document {documentId} from Vector Store!")
        print(e)
    
    loggingConfig["loggingObject"].info(f"Document {documentId} deleted Successfully from Vector Store.")
    return True


'''
INFERENCE V2
'''

def inference_v2(question, retriever, loggingConfig, inferenceId):
    
    RETRIEVED_AMM_1 = 3
    RETRIEVED_AMM_2 = 3

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
 
    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Pipeline initiated. User Prompt: {question}")
    prompt_1 = ChatPromptTemplate.from_template(template_1)

    llm_1 = OllamaLLM(model="llama3.1")

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Retrieval 1 started.")
    retrieved_docs_1_raw = retriever.invoke(question)
    retrieved_docs_1 = []
    for docs in retrieved_docs_1_raw:
        retrieved_docs_1.append(docs.page_content)
    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Retrieval 1 completed.")

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Reranking 1 started.")
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
            

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Reranking 1 completed.")


    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Augment 1 started.")
    augmented_prompt_1 = prompt_1.invoke({
        "context": context_1,
        "question": question
    }).to_string()
    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Augment 1 completed.")

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Generate 1 started.")
    generate_chain_1 =  llm_1 | StrOutputParser()
    response_1 = generate_chain_1.invoke(augmented_prompt_1)
    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Generate 1 completed.")

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
    """
    prompt_2 = ChatPromptTemplate.from_template(template_2)

    llm_2 = OllamaLLM(model="llama3.1")

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Retrieve 2 started.")
    retrieved_docs_2_raw = retriever.invoke(response_1)
    retrieved_docs_2 = []
    for docs in retrieved_docs_2_raw:
        retrieved_docs_2.append(docs.page_content)
    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Retrieve 2 completed.")

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Reranking 2 started.")
    context_2, docs_ranked_indices_2, docs_ranked_scores_2 = rerank(response_1, retrieved_docs_2)

    for score_raw in docs_ranked_scores_2[:RETRIEVED_AMM_2]:
        score = float(score_raw)
        retrieved_docs_ranked_scores.append(score)

    for i in docs_ranked_indices_2[:RETRIEVED_AMM_2]:
        
        id = retrieved_docs_2_raw[i].metadata['file_id']
        retrieved_docs_ranked_ids.append(id)

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Reranking 2 completed.")


    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Augment 2 started.")
    augmented_prompt_2 = prompt_2.invoke({
        "response_1": response_1,
        "context_1": context_1,
        "context_2": context_2,
        "question": question
        }).to_string()
    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Augment 2 completed.")

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Generate 2 started.")
    generate_chain_2 =  llm_2 | StrOutputParser()
    response_2 = generate_chain_2.invoke(augmented_prompt_2)
    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Generate 2 completed.")

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Pipeline completed.")
    
    contexts = {
        "ids": retrieved_docs_ranked_ids,
        "scores": retrieved_docs_ranked_scores
    }
    
    return response_2, augmented_prompt_1, augmented_prompt_2, contexts


'''
INFERENCE V3
'''

def inference_v3(question, retriever, loggingConfig, inferenceId):
    
    RETRIEVED_AMM = 3

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

    # Retrieval
    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Augment 1 start.")
    retrieved_docs_raw = retriever.invoke(question)
    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Augment 1 finished.")
    
    #  Prepare for Reranking
    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Rerankimg 1 start.")
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

    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Reranking 1 finished.")

    
    # Augment
    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Augment 1 start.")
    augmented_prompt = prompt.invoke({
        "context": retrieved_docs_ranked[:RETRIEVED_AMM],
        "question": question
    }).to_string()
    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Augment 1 finished.")

    # Generate
    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Generate 1 start.")
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
        loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Something went wrong when inferencing through the Azure AI Model Inference API.")
        print(e)

    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Generate 1 finished.")

    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Pipeline completed.")
    
                                     
    return response, augmented_prompt, contexts



def delete_doc_from_vectorstore(documentId, vectorStore, loggingConfig):
    ids = []
    try:
        ids.append(documentId)
        vectorStore.delete(ids=ids)
    except Exception as e:
        loggingConfig["loggingObject"].exception(f"Error when deleting Document {documentId} from Vector Store!")
        print(e)
    
    loggingConfig["loggingObject"].info(f"Document {documentId} deleted Successfully from Vector Store.")
    return True

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

def chat_v1(prompt, retriever, loggingConfig, inferenceId):

    # 1) Classify Prompt
    # 2) Generate Response

    # 1)
    classifications = {
        "cross-sell": "Question asks to perform Crosselling",
        "others": "Question does not ask to perform Cross-Selling"
    }
    response = questionClassifier(prompt, classifications, inferenceId, loggingConfig)
    
    if "cross-sell" in response:
        loggingConfig["loggingObject"].info(f"Inference {inferenceId} is classified as CROSS-SELL")
        response_raw, augmented_prompt, contexts = inference_v2(prompt, retriever, loggingConfig, inferenceId)
    else:
        loggingConfig["loggingObject"].info(f"Inference {inferenceId} is classified as OTHERS")
        response_raw, augmented_prompt_1, augmented_prompt_2, contexts = inference_v3(prompt, retriever, loggingConfig, inferenceId)
        augmented_prompt = augmented_prompt_2

    return response_raw, augmented_prompt, contexts

def questionClassifier(question, classifications, inferenceId, loggingConfig):
    
    classifications_list = "\n".join([f"- {key}: {value}" for key,value in classifications.items()])
    template ="""
Read the given Question and Answer only the Key of the Clasification from one of the following List of Classifications:
{classifications}

Example Question #1: What is GCP?
Response #1: others

Example Question #2: What can I recommend to Company ABC?
Response #2: cross-sell

Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    final_prompt = prompt.invoke({'question': question, 'classifications': classifications_list}).to_string()
    #  print(final_prompt)

    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Question Classification start.")
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
        loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Something went wrong when performing Questin Classification through the Azure AI Model Inference API.")
        print(e)

    loggingConfig["loggingObject"].info(f"[V3 | {inferenceId}] Question Classification completed.")
    return response