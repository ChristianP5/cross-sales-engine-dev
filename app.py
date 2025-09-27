from flask import Flask,render_template

import os
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from langchain_ollama import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document

import psycopg2

'''
================================================================================================
CONSTANTS START
'''
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

'''
INIT END
================================================================================================
'''

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

    return true

def pdf_to_postgresql(filename, fileId):

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        createDocsTable_sql_query = "CREATE TABLE IF NOT EXISTS document(id INT PRIMARY KEY, name VARCHAR(255), date VARCHAR(255))"
        cursor.execute(createDocsTable_sql_query)

        currentDate = ""
        insertDocEntry_sql_query = f"INSERT INTO document(id, name, date) VALUES ({fileId, filename, currentDate})"
        cursor.execute(insertDocEntry_sql_query)
 
    except Exception as e:
        print("Error when storing PDF to PostgreSQL Database!")
        print(e)
    finally:
        cursor.close()
        conn.close()
        


    return true

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


@app.route('/upload', methods=['GET'])
def upload_file_page():
    return render_template("upload.html")


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
        }

    # Upload the PDF Input
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    # Add Upload to PostgreSQL Database
    fileId = str(uuid4())[:8]
    # pdf_to_postgresql(filename, fileId)
    
    # Add PDF Input to Vector Store
    # pdf_to_vectorstore(filepath, fileId)

    # Build Response
    return {
        "status": "success",
        "message": "Upload Success!",
        "data": {
            "fileId": fileId
        }
    }

'''
API SERVER END
================================================================================================
'''

app.run(host="0.0.0.0", port=5000)