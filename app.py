from flask import Flask,render_template,send_from_directory

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

from vectorstore_utils import allowed_file, pdf_to_vectorstore, inference_v1, delete_doc_from_vectorstore, inference_v2
from psql_utills import get_db_connection, pdf_to_postgresql, list_documents, getInferencesById, getDocumentById, savePrompt_v1 ,savePrompt_v2 ,delete_doc_from_postgresql
from common_utils import generateId

def deleteDocumentById(filename, documentId):
    delete_doc_from_postgresql(documentId, PSQL_CONNECTION, LOGGING_CONFIGURATION)
    delete_doc_from_filesystem(filename)
    delete_doc_from_vectorstore(documentId, vector_store, LOGGING_CONFIGURATION)
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

    logging.info(f"File {filename} deleted Successfully from File Storage.")
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
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)

# Initialize Database
try:
    conn = get_db_connection(PSQL_CONNECTION)
    cursor = conn.cursor()
    createDocsTable_sql_query = "CREATE TABLE IF NOT EXISTS documents(documentId VARCHAR(255) PRIMARY KEY, name VARCHAR(255), type VARCHAR(255), dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createDocsTable_sql_query)

    createInferencesTable_sql_query = "CREATE TABLE IF NOT EXISTS inferences(inferenceId VARCHAR(255) PRIMARY KEY, userId VARCHAR(255), chatId VARCHAR(255), initialPrompt TEXT, finalPrompt TEXT, response TEXT, dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createInferencesTable_sql_query)

    createUsersTable_sql_query = "CREATE TABLE IF NOT EXISTS users(userId VARCHAR(255) PRIMARY KEY, username VARCHAR(255), password VARCHAR(255), dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createUsersTable_sql_query)

    createChatsTable_sql_query = "CREATE TABLE IF NOT EXISTS chats(chatId VARCHAR(255) PRIMARY KEY, userId VARCHAR(255), dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createChatsTable_sql_query)

    createInferencesV2Table_sql_query = "CREATE TABLE IF NOT EXISTS inferencesV2(inferenceId VARCHAR(255) PRIMARY KEY, userId VARCHAR(255), chatId VARCHAR(255), initialPrompt TEXT, finalPrompt1 TEXT, finalPrompt2 TEXT, response TEXT, dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createInferencesV2Table_sql_query)

    conn.commit()

    logging.info("Database Server Initalized")
except Exception as e:
    logging.exception("Error when Initializing Database")
    print(e)
finally:
    cursor.close()
    conn.close()


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
    return render_template("inventory.html")

@app.route('/chat', methods=['GET'])
def chat_page():
    return render_template("chat.html")

@app.route('/chats/<chatId>', methods=['GET'])
def chat_page_byId(chatId):
    return render_template("chatById.html")

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
    file = request.files['file']

    # Validate Input
    if not file or not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return {
            "status": "fail",
            "message": "Invalid Input"
        },400
    
    filename = secure_filename(file.filename)

    # Upload the PDF Input to File Storage
    logging.info(f"{filename} saving to File Storage")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    logging.info(f"{filename} saved to File Storage")

    # Add Upload to PostgreSQL Database
    logging.info(f"{filename} Metadata saving to Database")
    fileId = str(uuid4())[:8]
    pdf_to_postgresql(filename, fileId, PSQL_CONNECTION)
    logging.info(f"{filename} Metadata saved to Database")
    
    # Add PDF Input to Vector Store
    logging.info(f"{filename} Embedding saving to Vector Store")
    chunk_count = pdf_to_vectorstore(filepath, fileId, vector_store)
    logging.info(f"{filename}'s ({chunk_count} Chunks) Embedding saved to Vector Store")

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

    docs = list_documents(PSQL_CONNECTION, LOGGING_CONFIGURATION)

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

    doc = getDocumentById(documentId, PSQL_CONNECTION, LOGGING_CONFIGURATION)
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
    chatId = "TEST_CHAT"

    prompt = req.get('question')
    logging.info(f"Prompt received: {prompt}")

    augmented_prompt, response_raw = inference_v1(prompt, retriever)

    savePrompt_v1(userId, chatId, prompt, augmented_prompt, response_raw, PSQL_CONNECTION, LOGGING_CONFIGURATION)

    response_html = markdown.markdown(response_raw)

    return {
        "status": "success",
        "message": "Inference Successfully!",
        "data": {
            "user_prompt": prompt,
            "final_prompt": augmented_prompt,
            "response": response_raw,
            "response_html": response_html,
        }
    }

'''
INFERENCE V2 FEATURE
'''
@app.route('/v2/inference', methods=['POST'])
def post_inference_v2():

    req = request.get_json()

    userId = "TEST_USER"
    chatId = "TEST_CHAT"

    prompt = req.get('question')
    inferenceId = generateId(8)

    response_raw, augmented_prompt_1, augmented_prompt_2 = inference_v2(prompt, retriever, LOGGING_CONFIGURATION, inferenceId)

    # savePrompt_v2(userId, chatId, prompt, augmented_prompt, response_raw, PSQL_CONNECTION, LOGGING_CONFIGURATION, inferenceId)

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
        }
    }


'''
LIST INFERENCES By ID Feature
'''
@app.route('/v1/chats/<chatId>/inferences', methods=['GET'])
def get_inferences_byChatId(chatId):

    inferences = getInferencesById(chatId, PSQL_CONNECTION, LOGGING_CONFIGURATION)

    return {
        "status": "success",
        "message": "Inference Successfully!",
        "data": {
            "inferences": inferences
        }
    }

    
'''
API SERVER END
================================================================================================
'''
logging.info("Web Server Initalized")
app.run(host="0.0.0.0", port=80)
logging.info("Web Server Stopped")