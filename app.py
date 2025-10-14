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

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

'''
================================================================================================
UTILS START
'''
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_vectorstore(filepath, fileId):

    loader = PyPDFLoader(filepath)
    
    docs = loader.load()
    ids = [str(uuid4()) for _ in range(len(docs))]

    # For adding metadata to each chunk
    for doc in docs:
        doc.metadata["source_file"] = filepath
        doc.metadata["file_id"] = str(fileId)


    vector_store.add_documents(documents=docs, ids=ids)

    return len(docs)

def pdf_to_postgresql(filename, fileId):

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        currentDate = datetime.now()
        insertDocEntry_sql_query = f"INSERT INTO documents(documentId, name, type, dateCreated) VALUES (%s, %s, 'PDF', %s);"
        cursor.execute(insertDocEntry_sql_query, (fileId, filename, currentDate))

        conn.commit()
 
    except Exception as e:
        print("Error when storing PDF to PostgreSQL Database!")
        print(e)
    finally:
        cursor.close()
        conn.close()

    return True
        
def list_documents():
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        sql_query = "SELECT * FROM documents;"

        cursor.execute(sql_query)

        conn.commit()

        data = cursor.fetchall()
        docs = []
        for doc in data:
            item = {"id": doc[0], "name": doc[1], "type": doc[2], "date": doc[3]}
            docs.append(item)
    
    except Exception as e:
        logging.exception("Error when Listing Documents to PostgreSQL Database!")
        print(e)

    finally:
        cursor.close()
        conn.close()

    return docs


def get_db_connection():
    return psycopg2.connect(
        dbname=PSQL_DBNAME,
        user=PSQL_USERNAME,
        password=PSQL_PASSWORD,
        host=PSQL_HOST,
        port=PSQL_PORT
    )

def inference(question):
    
    template ="""You are an expert assistant answering user questions.
    Use the provided context as supporting information, but do not limit yourself strictly to it.
    If the context contains useful facts, include them.
    If the context does not fully answer the question, state that first, and then use your own general knowledge to give the best possible answer.
    But if the Context does not contain any relevant information, state that first, and then explain based on your general knowledge in less than 5 sentences.
    Always be accurate and clear.
Context: {context}
Question: {question}"""
 
    prompt = ChatPromptTemplate.from_template(template)

    llm = OllamaLLM(model="llama3.1")

    

    augment_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt

    augmented_prompt = augment_chain.invoke(question).to_string()

    generate_chain =  llm | StrOutputParser()
    response = generate_chain.invoke(augmented_prompt)

    return augmented_prompt, response

def savePrompt(userId, chatId, initialPrompt, finalPrompt, response):
    currentDate = datetime.now()
    inferenceId = hashlib.sha256(str(currentDate).encode()).hexdigest()[:8]

    try:
        logging.info("Saving Prompt Started.")
        conn = get_db_connection()
        cursor = conn.cursor()

        sql_query = f"INSERT INTO inferences(inferenceId, initialPrompt, finalPrompt, dateCreated, response, userId, chatId) VALUES (%s, %s, %s, %s, %s, %s, %s);"

        cursor.execute(sql_query, (inferenceId, initialPrompt, finalPrompt, currentDate, response, userId, chatId))

        conn.commit()
    
    except Exception as e:
        logging.exception("Error when Saving Prompt to PostgreSQL Database!")
        print(e)

    finally:
        cursor.close()
        conn.close()
        
    logging.info("Saving Prompt Finished.")
    return True

def getInferencesById(chatId):

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        sql_query = "SELECT * FROM inferences WHERE chatId = %s;"

        cursor.execute(sql_query, (chatId,))

        conn.commit()

        data = cursor.fetchall()

        # print(data)

        inferences = []
        
        for inference in data:
            item = {"inferenceId": inference[0], "userId": inference[1], "chatId": inference[2], "initialPrompt": inference[3], "finalPrompt": inference[4], "response": inference[5], "dateCreated": inference[6]}
            inferences.append(item)
        
    
    except Exception as e:
        logging.exception("Error when Listing Inferences to PostgreSQL Database!")
        print(e)

    finally:
        cursor.close()
        conn.close()

    return inferences

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
    conn = get_db_connection()
    cursor = conn.cursor()
    createDocsTable_sql_query = "CREATE TABLE IF NOT EXISTS documents(documentId VARCHAR(255) PRIMARY KEY, name VARCHAR(255), type VARCHAR(255), dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createDocsTable_sql_query)

    createInferencesTable_sql_query = "CREATE TABLE IF NOT EXISTS inferences(inferenceId VARCHAR(255) PRIMARY KEY, userId VARCHAR(255), chatId VARCHAR(255), initialPrompt TEXT, finalPrompt TEXT, response TEXT, dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createInferencesTable_sql_query)

    createUsersTable_sql_query = "CREATE TABLE IF NOT EXISTS users(userId VARCHAR(255) PRIMARY KEY, username VARCHAR(255), password VARCHAR(255), dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createUsersTable_sql_query)

    createChatsTable_sql_query = "CREATE TABLE IF NOT EXISTS chats(chatId VARCHAR(255) PRIMARY KEY, userId VARCHAR(255), dateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    cursor.execute(createChatsTable_sql_query)

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
    if not file or not allowed_file(file.filename):
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
    pdf_to_postgresql(filename, fileId)
    logging.info(f"{filename} Metadata saved to Database")
    
    # Add PDF Input to Vector Store
    logging.info(f"{filename} Embedding saving to Vector Store")
    chunk_count = pdf_to_vectorstore(filepath, fileId)
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

    docs = list_documents()

    return {
        "status": "success",
        "message": "Data Retrieved Successfully!",
        "data": {
            "docs": docs
        }
    }

'''
INFERENCE FEATURE
'''
@app.route('/v1/inference', methods=['POST'])
def post_inference():

    req = request.get_json()

    userId = "TEST_USER"
    chatId = "TEST_CHAT"

    prompt = req.get('question')
    logging.info(f"Prompt received: {prompt}")

    augmented_prompt, response = inference(prompt)

    savePrompt(userId, chatId, prompt, augmented_prompt, response)

    return {
        "status": "success",
        "message": "Inference Successfully!",
        "data": {
            "user_prompt": prompt,
            "final_prompt": augmented_prompt,
            "response": response,
        }
    }

'''
LIST INFERENCES By ID Feature
'''
@app.route('/v1/chats/<chatId>/inferences', methods=['GET'])
def get_inferences_byId(chatId):

    inferences = getInferencesById(chatId)

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