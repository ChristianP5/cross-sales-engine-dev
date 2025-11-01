import psycopg2
from datetime import datetime

import hashlib

import markdown

def get_db_connection(psqlConnectionConfig):
    
    return psycopg2.connect(
        dbname=psqlConnectionConfig["psqlDB"],
        user=psqlConnectionConfig["psqlUsername"],
        password=psqlConnectionConfig["psqlPass"],
        host=psqlConnectionConfig["psqlHost"],
        port=psqlConnectionConfig["psqlPort"]
    )

def pdf_to_postgresql(filename, fileId, psqlConnectionConfig):

    try:
        conn = get_db_connection(psqlConnectionConfig)
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


def list_documents(psqlConnectionConfig, loggingConfig):
    
    try:
        conn = get_db_connection(psqlConnectionConfig)
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
        loggingConfig["loggingObject"].exception("Error when Listing Documents to PostgreSQL Database!")
        print(e)

    finally:
        cursor.close()
        conn.close()

    return docs


def savePrompt(userId, chatId, initialPrompt, finalPrompt, response, psqlConnectionConfig, loggingConfig):
    currentDate = datetime.now()
    inferenceId = hashlib.sha256(str(currentDate).encode()).hexdigest()[:8]

    try:
        loggingConfig["loggingObject"].info("Saving Prompt Started.")
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = f"INSERT INTO inferences(inferenceId, initialPrompt, finalPrompt, dateCreated, response, userId, chatId) VALUES (%s, %s, %s, %s, %s, %s, %s);"

        cursor.execute(sql_query, (inferenceId, initialPrompt, finalPrompt, currentDate, response, userId, chatId))

        conn.commit()
    
    except Exception as e:
        loggingConfig["loggingObject"].exception("Error when Saving Prompt to PostgreSQL Database!")
        print(e)

    finally:
        cursor.close()
        conn.close()
        
    loggingConfig["loggingObject"].info("Saving Prompt Finished.")
    return True


def getInferencesById(chatId, psqlConnectionConfig, loggingConfig):

    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM inferences WHERE chatId = %s;"

        cursor.execute(sql_query, (chatId,))

        conn.commit()

        data = cursor.fetchall()

        # print(data)

        inferences = []
        
        for inference in data:
            response_raw = inference[5]
            response_html = markdown.markdown(response_raw)
            item = {"inferenceId": inference[0], "userId": inference[1], "chatId": inference[2], "initialPrompt": inference[3], "finalPrompt": inference[4], "response_raw": inference[5], "dateCreated": inference[6], "response_html": response_html}
            inferences.append(item)
        
    
    except Exception as e:
        loggingConfig["loggingObject"].exception("Error when Listing Inferences to PostgreSQL Database!")
        print(e)

    finally:
        cursor.close()
        conn.close()

    return inferences


def getDocumentById(documentId, psqlConnectionConfig, loggingConfig):
    
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM documents WHERE documentId = %s;"

        cursor.execute(sql_query, (documentId,))

        conn.commit()

        data = cursor.fetchall()

        docs = []
        for doc in data:
            item = {"id": doc[0], "name": doc[1], "type": doc[2], "date": doc[3]}
            docs.append(item)
    
    except Exception as e:
        loggingConfig["loggingObject"].exception(f"Error when retrieving Document {documentId} from PostgreSQL Database!")
        print(e)

    finally:
        cursor.close()
        conn.close()

    return docs[0]


def delete_doc_from_postgresql(documentId, psqlConnectionConfig, loggingConfig):
    try:
        loggingConfig["loggingObject"].info(f"Deleting Document {documentId} from PostgreSQL Database.")
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = f"DELETE FROM documents WHERE documentId = %s;"

        cursor.execute(sql_query, (documentId, ))

        conn.commit()
    
    except Exception as e:
        loggingConfig["loggingObject"].exception(f"Error when deleting Document {documentId} PostgreSQL Database!")
        print(e)
    finally:
        cursor.close()
        conn.close()
        
    loggingConfig["loggingObject"].info(f"Document {documentId} deleted Successfully from PostgreSQL Database.")

    return True