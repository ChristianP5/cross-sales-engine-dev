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

        currentDate = str(datetime.now())
        insertDocEntry_sql_query = f"INSERT INTO document(id, name, type, date) VALUES ('{fileId}', '{filename}', 'PDF', '{currentDate}');"
        cursor.execute(insertDocEntry_sql_query)

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

        sql_query = "SELECT * FROM document;"

        cursor.execute(sql_query)

        conn.commit()

        data = cursor.fetchall()
        docs = []
        for doc in data:
            item = {"id": doc[0], "type": doc[1], "name": doc[2], "date": doc[3]}
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
    createDocsTable_sql_query = "CREATE TABLE IF NOT EXISTS document(id VARCHAR(255) PRIMARY KEY, name VARCHAR(255), type VARCHAR(255), date VARCHAR(255));"
    cursor.execute(createDocsTable_sql_query)

    conn.commit()
except Exception as e:
    logging.exception("Error when Initializing Database")
    print(e)
finally:
    cursor.close()
    conn.close()





# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
UPLOAD FEATURE
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
API SERVER END
================================================================================================
'''
logging.info("Web Server Initalized")
app.run(host="0.0.0.0", port=80)
logging.info("Web Server Stopped")