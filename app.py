from flask import Flask,render_template,send_from_directory
import requests

import os
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from langchain_ollama import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document

import psycopg2
from datetime import datetime

import logging
import hashlib
import markdown

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

'''
================================================================================================
UTILS START
'''

import vectorstore_utils as vectorStoreUtils
import psql_utills as psqlUtils
from common_utils import generateId

def deleteDocumentById(filename, documentId):
    psqlUtils.delete_doc_from_postgresql(documentId, PSQL_CONNECTION, LOGGING_CONFIGURATION)
    delete_doc_from_filesystem(filename)
    vectorStoreUtils.delete_doc_from_vectorstore(documentId, vector_store, LOGGING_CONFIGURATION)
    return True


def delete_doc_from_filesystem(filename):

    try:
        # Upload the PDF Input to File Storage
        logging.info(f"{filename} deleting from File Storage")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.path.exists(filepath)
        os.remove(filepath)
    except Exception:
        logging.exception(f"Error when deleting File {filename} from File Storage!")
        print(e)
        return True

    logging.info(f"File {filename} deleted Successfully from File Storage.")
    return True

def saveInference_ChatV1(userId, chatId, initialPrompt, finalPrompt, response, psqlConnectionConfig, loggingConfig, context_ids, context_scores, inferenceId):
    loggingConfig["loggingObject"].info(f"[Chat V1 | Inference: {inferenceId}] Saving Prompt Pipeline Started.")
    
    try:
        # Save Inference to PostgreSQL
        psqlUtils.saveInference_ChatV1_to_postgresql(userId, chatId, initialPrompt, finalPrompt, response, psqlConnectionConfig, loggingConfig, inferenceId, context_ids, context_scores)

        # Save Response as Memory to Vector Store
        vectorStoreUtils.memory_to_vectorstore(inferenceId, initialPrompt, response, chatId, vector_store, LOGGING_CONFIGURATION)
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat V1 | Inference: {inferenceId}] Saving Prompt Pipeline Failed.")
        print(e)
        return True
        

    loggingConfig["loggingObject"].info(f"[Chat V1 | Inference: {inferenceId}] Saving Prompt Pipeline Successful.")
    return True

'''
UTILS END
================================================================================================
'''


'''
================================================================================================
CONSTANTS START
'''
PUBLIC_FILES_DIR = './public'
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
CHROMA_COLLECTION = 'knowledgebase_documents'
CHROMA_PERSIST_DIR = './chroma_langchain_db'
PSQL_DBNAME = "cross-sales-engine-dev-db1"
PSQL_HOST = "localhost"
PSQL_PORT = 5432
PSQL_USERNAME = "postgres"
PSQL_PASSWORD = "toor"

PSQL_CONNECTION = {
    "psqlDB": PSQL_DBNAME,
    "psqlUsername": PSQL_USERNAME,
    "psqlPass": PSQL_PASSWORD,
    "psqlHost": PSQL_HOST,
    "psqlPort": PSQL_PORT
}

LOGGING_CONFIGURATION = {
    "loggingObject": logging
}

OLLAMA_SERVER_CONF = {
    "base_url": "http://localhost:11434"
}




'''
CONSTANTS END
================================================================================================
'''

'''
================================================================================================
INIT START
'''

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Initialize Web Server
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Vector Store
embeddings = OllamaEmbeddings(model="mxbai-embed-large", keep_alive=-1)
vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={
            "k": 5,
            "fetch_k": 10,
            "filter": {
                "type": "document"
            }
        }
)



# Initialize Database
try:
    conn = psqlUtils.get_db_connection(PSQL_CONNECTION)
    cursor = conn.cursor()

    createInferencesTable_sql_query = "CREATE TABLE IF NOT EXISTS inferences(inferenceId VARCHAR(255) PRIMARY KEY, userId VARCHAR(255), chatId VARCHAR(255), initialPrompt TEXT, finalPrompt TEXT, response TEXT, dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP, context_ids TEXT[], context_scores REAL[]);"
    cursor.execute(createInferencesTable_sql_query)
   
    # for User Management
    createUsersTable_sql_query = "CREATE TABLE IF NOT EXISTS users(userId VARCHAR(255) PRIMARY KEY, username VARCHAR(255), password VARCHAR(255), dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createUsersTable_sql_query)

    # for Chat Management
    createChatsTable_sql_query = "CREATE TABLE IF NOT EXISTS chats(chatId VARCHAR(255) PRIMARY KEY, userId VARCHAR(255), name VARCHAR(255), dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createChatsTable_sql_query)

     # for Saving Inferences for InferenceV2
    createInferencesV2Table_sql_query = "CREATE TABLE IF NOT EXISTS inferencesV2(inferenceId VARCHAR(255) PRIMARY KEY, userId VARCHAR(255), chatId VARCHAR(255), initialPrompt TEXT, finalPrompt1 TEXT, finalPrompt2 TEXT, response TEXT, dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP, context_ids TEXT[], context_scores REAL[]);"
    cursor.execute(createInferencesV2Table_sql_query)

    # for Saving Inferences for ChatV1
    createInferencesChatV1Table_sql_query = "CREATE TABLE IF NOT EXISTS inferences_ChatV1(inferenceId VARCHAR(255) PRIMARY KEY, userId VARCHAR(255), chatId VARCHAR(255), initialPrompt TEXT, finalPrompt TEXT, response TEXT, dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP, context_ids TEXT[], context_scores REAL[]);"
    cursor.execute(createInferencesChatV1Table_sql_query)

    # for Document Management
    createDocsTable_sql_query = "CREATE TABLE IF NOT EXISTS documents(documentId VARCHAR(255) PRIMARY KEY, name TEXT, type VARCHAR(255), purpose TEXT, createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createDocsTable_sql_query)

    # for Customer Management
    createCustomersTable_sql_query = "CREATE TABLE IF NOT EXISTS customers(customerId TEXT PRIMARY KEY, name TEXT, profile TEXT, products TEXT, contacts TEXT, createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createCustomersTable_sql_query)

    conn.commit()

    logging.info("Database Server Initalized")
except Exception as e:
    logging.exception("Error when Initializing Database")
    print(e)
finally:
    cursor.close()
    conn.close()

# Initialize Ollama Server
try:
    llm = OllamaLLM(model='llama3.1', keep_alive=-1)
    llm.invoke("")
    logging.info("Ollama Server Initalized")
except Exception as e:
    logging.exception("Error when Initializing Ollama Server")
    print(e)

'''
INIT END
================================================================================================
'''

'''
================================================================================================
FRONTEND SERVER START
'''

@app.route('/inventory', methods=['GET'])
def upload_file_page():
    return render_template("inventory.html", active_page='inventory')

@app.route('/customers', methods=['GET'])
def customer_management_page():
    return render_template("customers.html", active_page='customers')

@app.route('/customers/<customerId>', methods=['GET'])
def customer_profile_page(customerId):
    return render_template("customerPage.html")

@app.route('/chat', methods=['GET'])
def chat_page():
    return render_template("chat.html")

@app.route('/chats/<chatId>', methods=['GET'])
def chat_page_byId(chatId):
    return render_template("chatById.html", active_page='chats')

@app.route('/file/<path:filename>', methods=['GET'])
def get_file(filename):
    return send_from_directory(PUBLIC_FILES_DIR, filename)



'''
FRONTEND SERVER END
================================================================================================
'''

'''
================================================================================================
API SERVER START
'''

'''
UPLOAD FEATURE
'''
@app.route('/v1/upload', methods=['POST'])
def upload_file():
    
    
    # Extract PDF from Request
    file = request.files.get('file')

    # Extarct fields from Request
    purpose = request.form.get('purpose')

    # If Purpose != REGULATION, verify if Customer ID exists
    if purpose != 'REGULATION':
        if not psqlUtils.getCustomerById(purpose, PSQL_CONNECTION, LOGGING_CONFIGURATION):
            return {
                "status": "fail",
                "message": f"Invalid Customer: {purpose}"
            },400
        

    # Validate Input
    if not file or not vectorStoreUtils.allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return {
            "status": "fail",
            "message": "Invalid Input"
        },400
    
    filename = secure_filename(file.filename)
    logging.info(f"Received {filename} with Purpose: {purpose}")

    # Upload the PDF Input to File Storage
    logging.info(f"{filename} saving to File Storage")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    logging.info(f"{filename} saved to File Storage")

    # Add Upload to PostgreSQL Database
    logging.info(f"{filename} Metadata saving to Database")
    fileId = str(uuid4())[:8]
    psqlUtils.pdf_to_postgresql(filename, fileId, PSQL_CONNECTION, purpose)
    logging.info(f"{filename} Metadata saved to Database")
    
    # Add PDF Input to Vector Store
    logging.info(f"{filename} Embedding saving to Vector Store")
    chunk_count = vectorStoreUtils.pdf_to_vectorstore(filepath, fileId, vector_store, purpose)
    logging.info(f"{filename}'s ({chunk_count} Chunks) Embedding saved to Vector Store")

    
    # If Purpose != REGULATION, Generate and Update Content for the Customer Information
    if purpose != 'REGULATION':

        # Profile
        profile_instructionConfig = {
            "role": "You are an enterprise solutions analyst and customer intelligence specialist.",
            "objective": """
            Generate a concise, accurate summary of a customer based strictly on the provided retrieved context. Behavior Rules:
            • Use ONLY the information explicitly present in the context.
            • Do NOT infer, assume, or hallucinate missing details.
            • If critical information is missing, explicitly state what is unavailable.
            • Maintain a professional, factual, and neutral tone suitable for internal business documentation.
            • Prefer bullet points for clarity when appropriate.
            """,
            "examples": """
            ### Customer Overview
            - **Customer Name:** PT Example Teknologi
            - **Industry:** Financial Services
            - **Segment:** Enterprise

            ### Relationship Summary
            - Currently using **Google Cloud Platform (GCP)** for core infrastructure
            - Previously engaged for **data analytics advisory**

            ### Technical Landscape
            - **Infrastructure:** GCP (Compute Engine, Cloud Storage)
            - **Data Platform:** BigQuery
            - **AI Usage:** Not mentioned in the provided context

            ### Business Needs & Challenges
            - Requires **scalable infrastructure** for seasonal workload spikes
            - Emphasis on **regulatory compliance** and **data security**

            ### Data Gaps
            - No information on current AI initiatives
            - No details on contract duration or commercial engagement

            """,
            "outputFormat":"""
            Use clear section headers and concise bullet points.
            """
        }

        profile = vectorStoreUtils.generateCustomerProfile(purpose, vector_store, LOGGING_CONFIGURATION, profile_instructionConfig)

        # print(f"profile: {profile}")

        profile_html = markdown.markdown(profile)

        psqlUtils.updateCustomer(purpose, PSQL_CONNECTION, LOGGING_CONFIGURATION, 'profile', profile_html)
    
        # Products
        products_instructionConfig = {
            "role": "You are an enterprise customer intelligence analyst specializing in product usage identification.",
            "objective": """
            Extract a list of product or service names that the customer is explicitly stated to be currently using, based strictly on the provided context.
            Behavior Rules:
            • Use ONLY product names that are explicitly mentioned in the context.
            • Do NOT infer, assume, or extrapolate usage.
            • Do NOT include products mentioned as historical, proposed, evaluated, or planned unless clearly stated as currently in use.
            • If no products are found, return an empty list.
            • Do NOT include explanations, commentary, or additional text.
            • Maintain exact product naming as written in the source context.
            """,
            "examples": """
            If Product Exists:
            - Google Cloud Platform
            - Compute Engine
            - Cloud Storage
            - BigQuery

            If No Product Exists:
            (empty)

            """,
            "outputFormat":"""
            Only output a List or an Empty List
            """
        }

        products = vectorStoreUtils.generateCustomerProfile(purpose, vector_store, LOGGING_CONFIGURATION, products_instructionConfig)

        # print(f"products: {products}")

        products_html = markdown.markdown(products)

        psqlUtils.updateCustomer(purpose, PSQL_CONNECTION, LOGGING_CONFIGURATION, 'products', products_html)
    
        # Contacts
        contacts_instructionConfig = {
            "role": "You are an enterprise account intelligence analyst specializing in extracting verified contact information.",
            "objective": """
            Extract a list of contact information related to the specified customer, including:
            • Customer-side Points of Contact (PIC)
            • PT Multipolar Technology Points of Contact (PIC)

            Behavior Rules:
            • Use ONLY contact information explicitly stated in the provided context.
            • Do NOT infer roles, names, or relationships.
            • Do NOT create or guess email addresses, phone numbers, or job titles.
            • Preserve the original wording and naming from the source context.
            • If no contacts are found for a category, return an empty list for that category.
            • Do NOT include explanations, assumptions, or commentary.
            """,
            "examples": """
            ### Customer PICs
            - Name: Andi Pratama
            Organization: PT Example Teknologi
            Role: IT Infrastructure Manager
            Email: andi.pratama@example.co.id
            Phone: +62 812-3456-7890

            ### PT Multipolar Technology PICs
            - Name: Rina Wijaya
            Organization: PT Multipolar Technology Tbk
            Role: Account Manager
            Email: rina.wijaya@multipolar.com
            Phone: Not mentioned
            """,
            "outputFormat":"""
            ### Customer PICs
            - Name:
            Organization:
            Role:
            Email:
            Phone:

            ### PT Multipolar Technology PICs
            - Name:
            Organization:
            Role:
            Email:
            Phone:
            """
        }

        contacts = vectorStoreUtils.generateCustomerProfile(purpose, vector_store, LOGGING_CONFIGURATION, contacts_instructionConfig)

        # print(f"contacts: {contacts}")

        contacts_html = markdown.markdown(contacts)

        psqlUtils.updateCustomer(purpose, PSQL_CONNECTION, LOGGING_CONFIGURATION, 'contacts', contacts_html)
    

    # Build Response
    logging.info(f"/v1/upload executed smoothly for {filename}")
    return {
        "status": "success",
        "message": "Upload Success!",
        "data": {
            "fileId": fileId
        }
    },201

'''
LIST DOCUMENTS FEATURE
'''
@app.route('/v1/documents', methods=['GET'])
def get_list_documents():

    docs = psqlUtils.list_documents(PSQL_CONNECTION, LOGGING_CONFIGURATION)

    return {
        "status": "success",
        "message": "Data Retrieved Successfully!",
        "data": {
            "docs": docs
        }
    }

'''
DELETE DOCUMENTS FEATURE
'''
@app.route('/v1/documents/<documentId>', methods=['DELETE'])
def delete_document_byId(documentId):

    doc = psqlUtils.getDocumentById(documentId, PSQL_CONNECTION, LOGGING_CONFIGURATION)
    filename = doc['name']

    print(f"Deletig Filename:{filename}")
    deleteDocumentById(filename, documentId)

    return {
        "status": "success",
        "message": f"Document {documentId} deleted successfully Successfully!",
    }

'''
INFERENCE V1 FEATURE
'''
@app.route('/v1/inference', methods=['POST'])
def post_inference_v1():

    req = request.get_json()

    userId = "TEST_USER"
    chatId = req.get('chatId')

    prompt = req.get('question')
    logging.info(f"Prompt received: {prompt}")

    augmented_prompt, response_raw, contexts = vectorStoreUtils.inference_v1(prompt, retriever)

    contexts_ids = contexts["ids"]
    contexts_scores = contexts["scores"]

    docs = []
    for id in contexts_ids:
        doc = psqlUtils.getDocumentById(id, PSQL_CONNECTION, LOGGING_CONFIGURATION)
        docs.append(doc)
    

    psqlUtils.savePrompt_v1(userId, chatId, prompt, augmented_prompt, response_raw, PSQL_CONNECTION, LOGGING_CONFIGURATION, contexts_ids, contexts_scores)

    response_html = markdown.markdown(response_raw)

    return {
        "status": "success",
        "message": "Inference Successfully!",
        "data": {
            "user_prompt": prompt,
            "final_prompt": augmented_prompt,
            "response": response_raw,
            "response_html": response_html,
            "docs": docs
        }
    }

'''
INFERENCE V2 FEATURE
'''
@app.route('/v2/inference', methods=['POST'])
def post_inference_v2():

    req = request.get_json()

    userId = "TEST_USER"
    chatId = req.get('chatId')

    prompt = req.get('question')
    inferenceId = generateId(8)
    logging.info(f"[V2 | Chat: {chatId}] Prompt received: {prompt} || Assigned Inference ID: {inferenceId}")

    response_raw, augmented_prompt_1, augmented_prompt_2, contexts = inference_v2(prompt, retriever, LOGGING_CONFIGURATION, inferenceId)

    contexts_ids = contexts["ids"]
    contexts_scores = contexts["scores"]

    docs = []
    for id in contexts_ids:
        doc = psqlUtils.getDocumentById(id, PSQL_CONNECTION, LOGGING_CONFIGURATION)
        docs.append(doc)
    

    psqlUtils.savePrompt_v2(userId, chatId, prompt, augmented_prompt_1, augmented_prompt_2, response_raw, PSQL_CONNECTION, LOGGING_CONFIGURATION, inferenceId, contexts_ids, contexts_scores)

    response_html = markdown.markdown(response_raw)

    return {
        "status": "success",
        "message": "Inference Successfully!",
        "data": {
            "user_prompt": prompt,
            "augmented_prompt_1": augmented_prompt_1,
            "augmeented_prompt_2": augmented_prompt_2,
            "response": response_raw,
            "response_html": response_html,
            "docs": docs
        }
    }

'''
INFERENCE V3 (Cloud) FEATURE
'''
@app.route('/v3/inference', methods=['POST'])
def post_inference_v3():

    req = request.get_json()

    userId = "TEST_USER"
    chatId = req.get('chatId')

    prompt = req.get('question')
    inferenceId = generateId(8)
    logging.info(f"[V3 | Chat: {chatId}] Prompt received: {prompt} || Assigned Inference ID: {inferenceId}")

    response_raw, augmented_prompt, contexts = vectorStoreUtils.inference_v3(prompt, retriever, LOGGING_CONFIGURATION, inferenceId)

    contexts_ids = contexts["ids"]
    contexts_scores = contexts["scores"]

    docs = []
    for id in contexts_ids:
        doc = psqlUtils.getDocumentById(id, PSQL_CONNECTION, LOGGING_CONFIGURATION)
        docs.append(doc)

    # savePrompt_v3(userId, chatId, prompt, augmented_prompt_1, augmented_prompt_2, response_raw, PSQL_CONNECTION, LOGGING_CONFIGURATION, inferenceId, contexts_ids, contexts_scores)
    
    
    response_html = markdown.markdown(response_raw)

    return {
        "status": "success",
        "message": "Inference Successfully!",
        "data": {
            "user_prompt": prompt,
            "augmented_prompt": augmented_prompt,
            "response": response_raw,
            "response_html": response_html,
            "docs": docs
        }
    }

'''
CHAT V1 FEATURE
'''
@app.route('/v1/chat', methods=['POST'])
def post_chat_v1():

    req = request.get_json()
    prompt = req.get('question')
    
    userId = "TEST_USER"
    chatId = req.get('chatId')

    
    inferenceId = generateId(8)
    logging.info(f"[Chat V1 | Chat: {chatId}] Prompt received: {prompt} || Assigned Inference ID: {inferenceId}")

    response_raw, augmented_prompt, contexts = vectorStoreUtils.chat_v1(prompt, retriever, LOGGING_CONFIGURATION, inferenceId, vector_store, chatId)

    contexts_ids = contexts["ids"]
    contexts_scores = contexts["scores"]

    docs = []
    for id in contexts_ids:
        doc = psqlUtils.getDocumentById(id, PSQL_CONNECTION, LOGGING_CONFIGURATION)
        docs.append(doc)
    
    # for Reviewing and Memory feature
    saveInference_ChatV1(userId, chatId, prompt, augmented_prompt,response_raw, PSQL_CONNECTION, LOGGING_CONFIGURATION, contexts_ids, contexts_scores, inferenceId)

    response_html = markdown.markdown(response_raw)

    return {
        "status": "success",
        "message": "Chat Successfully!",
        "data": {
            "user_prompt": prompt,
            "augmented_prompt": augmented_prompt,
            "response": response_raw,
            "response_html": response_html,
            "docs": docs
        }
    }

'''
LIST INFERENCES By Chat ID Feature
'''
@app.route('/v1/chats/<chatId>/inferences', methods=['GET'])
def get_inferences_byChatId(chatId):

    # Old Implementation
    # inferences = getInferencesById(chatId, PSQL_CONNECTION, LOGGING_CONFIGURATION)

    # New Implementation
    inferences = psqlUtils.getInference_ChatV1_from_postgresql(chatId, PSQL_CONNECTION, LOGGING_CONFIGURATION)


    return {
        "status": "success",
        "message": "Inference Successfully!",
        "data": {
            "inferences": inferences
        }
    }


'''
CHAT MANAGEMENT feature
'''
@app.route('/v1/chats', methods=['GET'])
def getChats():
    # Get Chat by User Id
    userId = 'TEST_USER'

    chats = psqlUtils.getChatsbyUserId(userId, PSQL_CONNECTION, LOGGING_CONFIGURATION)

    return {
        "status": "success",
        "message": f"Retrieved Chats of User {userId} Successfully!",
        "data": {
            "chats": chats
        }
    }

@app.route('/v1/chats/create', methods=['POST'])
def postChats():
    # Create Chat on User Id
    userId = 'TEST_USER'

    # Generate chatId
    chatId = generateId(8)

    # Get Display Name of the Chat
    req = request.get_json()
    name = req.get('name')

    psqlUtils.createChat(userId, PSQL_CONNECTION, LOGGING_CONFIGURATION, chatId, name)

    return {
        "status": "success",
        "message": f"Chats {chatId} Created for User {userId} Successfully!",
        "data": {
            "chatId": chatId
        }
    }

'''
CUSTOMER MANAGEMENT feature
'''
@app.route('/v1/customers/create', methods=['POST'])
def postCustomers():
    
    # Generate chatId
    customerId = generateId(8)

    # Get Properties of the Customer
    req = request.get_json()
    name = req.get('name')

    # Validate Input
    if name == '':
        return {
            "status": "fail",
            "message": "'name' cannot be empty"
        },400

    psqlUtils.createCustomer(customerId, PSQL_CONNECTION, LOGGING_CONFIGURATION, name)

    return {
        "status": "success",
        "message": f"Customer {customerId} Created Successfully!",
        "data": {
            "customerId": customerId
        }
    }

@app.route('/v1/customers', methods=['GET'])
def getCustomers():
  
    customers = psqlUtils.getCustomers(PSQL_CONNECTION, LOGGING_CONFIGURATION)

    return {
        "status": "success",
        "message": f"Customer retrieved Successfully!",
        "data": {
            "customers": customers
        }
    }

@app.route('/v1/customers/<customerId>', methods=['GET'])
def getCustomerById(customerId):
  
    customer = psqlUtils.getCustomerById(customerId, PSQL_CONNECTION, LOGGING_CONFIGURATION)

    return {
        "status": "success",
        "message": f"Customer {customerId} retrieved Successfully!",
        "data": {
            "customer": customer
        }
    }

@app.route('/v1/customers/<customerId>/documents', methods=['GET'])
def getDocuments_byCustomerById(customerId):
  
    docs = psqlUtils.getDocumentsByCustomerById(customerId, PSQL_CONNECTION, LOGGING_CONFIGURATION)

    return {
        "status": "success",
        "message": f"Documents for Customer {customerId} retrieved Successfully!",
        "data": {
            "docs": docs
        }
    }

'''
REGULATION MANAGEMENT feature
'''
@app.route('/v1/regulations', methods=['GET'])
def getRegulations():
  
    docs = psqlUtils.getRegulationDocs(PSQL_CONNECTION, LOGGING_CONFIGURATION)

    return {
        "status": "success",
        "message": f"Regulation Documents retrieved Successfully!",
        "data": {
            "docs": docs
        }
    }

@app.route('/v1/customerDocs', methods=['GET'])
def getCustomerDocs():
  
    docs = psqlUtils.getAllCustomerDocs(PSQL_CONNECTION, LOGGING_CONFIGURATION)

    return {
        "status": "success",
        "message": f"ALL Customer Documents retrieved Successfully!",
        "data": {
            "docs": docs
        }
    }

'''
API SERVER END
================================================================================================
'''
logging.info("Web Server Initalized")
app.run(host="0.0.0.0", port=80)
logging.info("Web Server Stopped")

requests.post(f"{OLLAMA_SERVER_CONF['base_url']}/api/generate", json={"model": "llama3.1", "keep_alive": 0})
requests.post(f"{OLLAMA_SERVER_CONF['base_url']}/api/generate", json={"model": "mxbai-embed-large", "keep_alive": 0})
logging.info("Ollama Server Stopped")