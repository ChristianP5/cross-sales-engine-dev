from langchain_community.document_loaders import PyPDFLoader
from uuid import uuid4

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap



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
    
    template ="""Role: You are an expert B2B sales strategist and solutions architect specializing in identifying cross-sell opportunities.
Goal: Based on the retrieved context, analyze the customer’s existing technology environment, company best practices, and vendor product portfolio to recommend additional products, services, or upgrades that align with the customer’s business goals and technology stack.
Context Provided: {context}
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

    

    augment_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt

    augmented_prompt = augment_chain.invoke(question).to_string()

    generate_chain =  llm | StrOutputParser()
    response = generate_chain.invoke(augmented_prompt)

    return augmented_prompt, response


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
    context_1 = retriever.invoke(question)
    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Retrieval 1 completed.")

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
    
    Retrieved Information/Context about our Customer: {context_1}
    Retrieved Information/Context about our Products: {context_2}
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
    Keep in mind that the original User's Question was: {question}
    """
    prompt_2 = ChatPromptTemplate.from_template(template_2)

    llm_2 = OllamaLLM(model="llama3.1")

    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Retrieve 2 started.")
    context_2 = retriever.invoke(response_1)
    loggingConfig["loggingObject"].info(f"[V2 | {inferenceId}] Retrieve 2 completed.")

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
    return response_2, augmented_prompt_1, augmented_prompt_2, 


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