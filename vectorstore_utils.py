from langchain_community.document_loaders import PyPDFLoader
from uuid import uuid4
import fitz             # from PyMuPDF
import PIL.Image        # from pillow
import io
import base64

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

    """
    1) Add Document to Vector Store
    2) Add AI-Generated Text of 'the Images inside Documents' to Vector Store
    """

    # 1)
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

    # 2)
    image_counter = 0

    pdf = fitz.open(filepath)

    for i in range(len(pdf)):
        page = pdf[i]
        images = page.get_images()

        for image in images:
            image_counter += 1
            base_img = pdf.extract_image(image[0])
            image_data = base_img["image"]

            img = PIL.Image.open(io.BytesIO(image_data))

            # Convert Image Data (bytes) to Base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            base64_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")
            image_data_url = f"data:image/png;base64,{base64_bytes}"

            # Get AI-Generated Summary of the Base64-encoded Image Data
            AZURE_ENDPOINT_UPLOADFEATURE = "https://c-ailab-aifoundry1.cognitiveservices.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
            AZURE_API_KEY_UPLOADFEATURE = os.getenv("AZURE_API_KEY_UPLOADFEATURE")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AZURE_API_KEY_UPLOADFEATURE}"
            }

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "You are converting images into text for a knowledge retrieval system.\n"
                                        "Describe the image in detail, including:\n"
                                        "- Image type\n"
                                        "- Entities and components\n"
                                        "- Relationships or flows\n"
                                        "- Labels, legends, axes, and units\n"
                                        "- Any visible text\n"
                                        "Be factual and structured."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url
                                    }
                                }
                            ]
                    }
                ],
                "max_completion_tokens": 30000,
                "model": "gpt-4.1"
            }

            response_raw = requests.post(AZURE_ENDPOINT_UPLOADFEATURE, headers=headers, json=payload)
            response = response_raw.json()["choices"][0]["message"]["content"]

            print(f"[Image #{image_counter}]: {response}\n")

            # Add AI-Generated Text to Vector Store
            page_content = f"{response}\n Page Number: {i+1}\n Image Index: {image_counter-1}"
    
            doc = Document(
                page_content=page_content,
            )

            doc.metadata["type"] = "document"
            doc.metadata["purpose"] = purpose
            doc.metadata["file_id"] = str(fileId)
            doc.metadata["source_file"] = filepath
            doc.metadata["derived"] = "Image"
            doc.metadata["page"] = i+1
            doc.metadata["index"] = image_counter-1

            try:
                id = str(uuid4())
                vectorStore.add_documents(documents=[doc], ids=[id])
                print(f"Image #{image_counter-1} from File {fileId} Page {i+1} is Aded to Vector Store.")

            except Exception as e:
                print(e)
                return True
 

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
    
    # print(f"final_prompt: {final_prompt}")

    
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
        response_raw, augmented_prompt_1, augmented_prompt_2, contexts = crossSellPipeline_chat_v2(prompt, retriever, loggingConfig, inferenceId, initial_retrieval_prompt, llm_memory_formatted, recent_llm_memory, regulationDocs_formatted, contexts)
        augmented_prompt = augmented_prompt_2

        '''
        print(f"\n\n=========================\nAugmented Prompt #1:")
        print(augmented_prompt_1)

        print(f"\n\n=========================\nAugmented Prompt #2:")
        print(augmented_prompt_2)

        print(f"\n\n=========================\n=========================\nResponse:")
        print(response_raw)
        '''

    else:
        loggingConfig["loggingObject"].info(f"[Chat V2 | Chat: {chatId} Inference: {inferenceId}] Inference is classified as OTHERS")
        response_raw, augmented_prompt, contexts = defaultPipeline_chat_v2(prompt, retriever, loggingConfig, inferenceId, llm_memory_formatted, chatId, initial_retrieval_prompt, recent_llm_memory, regulationDocs_formatted, contexts)

    return response_raw, augmented_prompt, contexts


def crossSellPipeline_chat_v2(question, retriever, loggingConfig, inferenceId, initial_retrieval_prompt, relevant_llm_memory, recent_llm_memory, regulationDocs, contexts):
    
    # Constants
    RETRIEVED_AMM_1 = 5
    RETRIEVED_AMM_2 = 5

    '''
    Prompt #1
    '''

    template_1 ="""Role:
You are an expert B2B sales strategist and solutions architect specializing in identifying compliant cross-sell opportunities.

Primary Goal:
Based on the retrieved customer context and applicable regulation documents, generate a list of recommended capabilities that align with the customer’s existing technology environment, company best practices, and vendor product portfolios.

---

SOURCE PRIORITY (DO NOT VIOLATE):
1. Regulation Documents (Mandatory Compliance Constraints)
2. Retrieved Customer Context / Technical Environment
3. Direct Recent Conversations
4. Relevant Past Answers (Reference Only)
5. General Industry Knowledge (only if clearly applicable)

If any lower-priority source conflicts with a higher-priority one:
• You MUST follow the higher-priority source
• You MUST explicitly state the compliance limitation

---

Regulation Documents (Retrieved — AUTHORITATIVE):
{regulation_context}

---

Retrieved Customer Context / Technical Environment:
{context}

---

Direct Recent Conversations (Most Recent First — USE FOR CONTEXTUAL CONTINUITY ONLY):
{recent_llm_memory}

---

Relevant Past Answers (REFERENCE ONLY — DO NOT INTRODUCE NEW FACTS):
{relevant_llm_memory}

---

Your Tasks:
1. Identify the customer’s current technology environment strictly from the Retrieved Customer Context
2. Identify compliance requirements, restrictions, or obligations imposed by the Regulation Documents
3. Identify gaps or opportunities where compliant capabilities could improve the customer’s environment
4. Define the capabilities recommended products or services MUST have to satisfy both:
   • Business goals
   • Regulatory obligations
5. For each recommended capability, provide:
   • Capability Description (up to 1 concise paragraph)

---

CRITICAL CONSTRAINTS:
• Regulations are mandatory — do NOT suggest non-compliant solutions
• Do NOT perform legal interpretation; only map capabilities to stated regulatory requirements
• Base all technical facts strictly on retrieved context
• Use memory ONLY to maintain continuity and avoid repetition
• Do NOT hallucinate unavailable data
• If regulation or customer data is insufficient, state what additional information is required
• Use concise, professional, sales-enablement language

---

Output Format (STRICT — DO NOT ADD EXTRA SECTIONS):

List of Recommended Capabilities:
- [RECOMMENDATION]: [Capability Description].  
  
---

Example Output:
List of Recommended Capabilities:
- A centralized log and audit data platform to retain, correlate, and analyze security events across infrastructure and applications, supporting incident investigation and long-term audit requirements.  
- A cloud-native data encryption and key management capability that enforces encryption at rest and in transit while maintaining centralized key control.  
  
---

Question:
{question}
"""
 
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Pipeline initiated. User Prompt: {question}")
    prompt_1 = ChatPromptTemplate.from_template(template_1)

    # Retrieval 1
    retrieval_question = initial_retrieval_prompt

    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Retrieval 1 started.")
    retrieved_docs_1_raw = retriever.invoke(retrieval_question)
    retrieved_docs_1 = []
    for docs in retrieved_docs_1_raw:
        retrieved_docs_1.append(docs.page_content)
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Retrieval 1 completed.")

    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Reranking 1 started.")
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
            

    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Reranking 1 completed.")

    # Add the Retrieved Context to the Contexts List
    for ids in retrieved_docs_ranked_ids:
        contexts["ids"].append(ids)
    
    for scores in retrieved_docs_ranked_scores:
        contexts["scores"].append(scores)

    # Augment 1
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Augment 1 started.")
    augmented_prompt_1 = prompt_1.invoke({
        "context": context_1,
        "question": question,
        "relevant_llm_memory": relevant_llm_memory,
        "recent_llm_memory": recent_llm_memory,
        "regulation_context": regulationDocs
    }).to_string()
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Augment 1 completed.")

    # Generate 1
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Generate 1 start.")
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
                "content": f"{augmented_prompt_1}"
            }
        ],
        "max_completion_tokens": 40000,
        "model": "o4-mini"
    }

    try:
        response_raw = requests.post(AZURE_ENDPOINT, headers=headers, json=payload)
        response_1 = response_raw.json()["choices"][0]["message"]["content"]
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Something went wrong when inferencing through the Azure AI Model Inference API.")
        print(e)

    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Generate 1 finished.")

    '''
    Prompt #2
    '''
    template_2 ="""Role:
You are an expert B2B sales strategist and solutions architect specializing in compliant cross-sell recommendations.

Background:
Previously, you were asked to recommend high-level capabilities that could improve the customer’s environment.
You already generated the following list of recommended capabilities:

Previously Recommended Capabilities (AUTHORITATIVE — DO NOT MODIFY):
{response_1}

You MUST:
• Use ONLY the capabilities listed above
• NOT introduce new capabilities
• NOT remove or merge existing capabilities

---

Primary Goal:
Explain WHY each previously recommended capability is relevant, and identify which of our available products or services can provide those capabilities, while ensuring alignment with customer needs and applicable regulations.

---

SOURCE PRIORITY (DO NOT VIOLATE):
1. Regulation Documents (Mandatory Compliance Constraints)
2. Previously Recommended Capabilities
3. Retrieved Customer Context
4. Retrieved Product Context
5. Direct Recent Conversations
6. Relevant Past Answers
7. General Knowledge (only if unavoidable and explicitly stated)

If any lower-priority source conflicts with a higher-priority source:
• Follow the higher-priority source
• Explicitly state the limitation

---

Regulation Documents (Retrieved — AUTHORITATIVE):
{regulation_context}

---

Retrieved Customer Information:
{context_1}

---

Retrieved Product / Service Information:
{context_2}

---

Direct Recent Conversations (Most Recent First — CONTEXT ONLY):
{recent_llm_memory}

---

Relevant Past Answers (REFERENCE ONLY):
{relevant_llm_memory}

---

Your Tasks:
1. Answer the Question:
{question}

2. For EACH capability listed in "Previously Recommended Capabilities", identify:
   • One or more matching products or services from the Retrieved Product Context
   • Why the product fits this customer’s environment and needs
   • The business or technical value delivered
   • Any dependencies, upgrade paths, or prerequisites
   • How to position this recommendation in a sales conversation

3. Explicitly ensure:
   • Product recommendations do NOT violate any Regulation Documents
   • If a capability cannot be mapped to a compliant product, state this clearly

---

CRITICAL CONSTRAINTS:
• Do NOT generate new capabilities
• Do NOT reinterpret regulations
• Do NOT hallucinate missing product features
• If data is insufficient, state what additional information is needed
• Use memory ONLY to avoid repetition or contradiction
• Use concise, professional language suitable for sales enablement
• Recommend a relevant Contact Person according to PT Multipolar Technology guidelines

---

Output Format (Markdown — STRICT):

### Brief Summary of Customer's Environment
[1 concise paragraph based strictly on retrieved customer context]

### Cross-Sell Recommendations

#### Capability: [Capability Name from "Previously Recommended Capabilities"]

##### [Product / Service Name]
* **Short Description:** …
* **Fit Rationale:** …
* **Value Proposition:** …
* **Dependencies / Prerequisites:** …
* **Confidence Level:** High / Medium / Low
* **Compliance Alignment:** [High-level reference to regulation]
* **Sales Positioning Tip:** …

(Repeat for each capability and product)
    """
    prompt_2 = ChatPromptTemplate.from_template(template_2)

    # Retrieve 2
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Retrieve 2 started.")
    retrieved_docs_2_raw = retriever.invoke(response_1)
    retrieved_docs_2 = []
    for docs in retrieved_docs_2_raw:
        retrieved_docs_2.append(docs.page_content)
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Retrieve 2 completed.")

    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Reranking 2 started.")
    context_2, docs_ranked_indices_2, docs_ranked_scores_2 = rerank(response_1, retrieved_docs_2)

    for score_raw in docs_ranked_scores_2[:RETRIEVED_AMM_2]:
        score = float(score_raw)
        retrieved_docs_ranked_scores.append(score)

    for i in docs_ranked_indices_2[:RETRIEVED_AMM_2]:
        
        id = retrieved_docs_2_raw[i].metadata['file_id']
        retrieved_docs_ranked_ids.append(id)

    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Reranking 2 completed.")

    # Add the Retrieved Context to the Contexts List
    for ids in retrieved_docs_ranked_ids:
        contexts["ids"].append(ids)
    
    for scores in retrieved_docs_ranked_scores:
        contexts["scores"].append(scores)

    # Augment 2
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Augment 2 started.")
    augmented_prompt_2 = prompt_2.invoke({
        "response_1": response_1,
        "context_1": context_1,
        "context_2": context_2,
        "question": question,
        "relevant_llm_memory": relevant_llm_memory,
        "recent_llm_memory": recent_llm_memory,
        "regulation_context": regulationDocs
        }).to_string()
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Augment 2 completed.")

    # Generate 2
    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Generate 2 started.")
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
                "content": f"{augmented_prompt_2}"
            }
        ],
        "max_completion_tokens": 100000,
        "model": "o4-mini"
    }

    try:
        response_raw = requests.post(AZURE_ENDPOINT, headers=headers, json=payload)
        response_2 = response_raw.json()["choices"][0]["message"]["content"]
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Something went wrong when inferencing through the Azure AI Model Inference API.")
        print(e)

    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Generate 2 finished.")

    loggingConfig["loggingObject"].info(f"[Chat V2 - cross-sell | Inference: {inferenceId}] Pipeline completed.")
    
    
    return response_2, augmented_prompt_1, augmented_prompt_2, contexts


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