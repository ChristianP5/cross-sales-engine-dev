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

def pdf_to_postgresql(filename, fileId, psqlConnectionConfig, purpose):

    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        currentDate = datetime.now()
        insertDocEntry_sql_query = f"INSERT INTO documents(documentId, name, type, createdAt, updatedAt, purpose) VALUES (%s, %s, 'PDF', %s, %s, %s);"
        cursor.execute(insertDocEntry_sql_query, (fileId, filename, currentDate, currentDate, purpose))

        conn.commit()
 
    except Exception as e:
        print(e)
        raise Exception("Error when storing PDF to PostgreSQL Database!")
        
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

        # expandPsqlOutput(data)

        docs = []
        for doc in data:
            item = {"id": doc[0], "name": doc[1], "type": doc[2], "purpose": doc[3], "createdAt": doc[4], "updatedAt": doc[5]}
            docs.append(item)
    
    except Exception as e:
        loggingConfig["loggingObject"].exception("Error when Listing Documents to PostgreSQL Database!")
        print(e)
        raise Exception("Error when Listing Documents to PostgreSQL Database!")

    finally:
        cursor.close()
        conn.close()

    return docs


def savePrompt_v1(userId, chatId, initialPrompt, finalPrompt, response, psqlConnectionConfig, loggingConfig, context_ids, context_scores):
    currentDate = datetime.now()
    inferenceId = hashlib.sha256(str(currentDate).encode()).hexdigest()[:8]

    try:
        loggingConfig["loggingObject"].info("Saving Prompt Started.")
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = f"INSERT INTO inferences(inferenceId, initialPrompt, finalPrompt, createdAt, response, userId, chatId, context_ids, context_scores) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"

        cursor.execute(sql_query, (inferenceId, initialPrompt, finalPrompt, currentDate, response, userId, chatId, context_ids, context_scores))

        conn.commit()
    
    except Exception as e:
        loggingConfig["loggingObject"].exception("Error when Saving Prompt to PostgreSQL Database!")
        print(e)
        raise Exception("Error when Saving Prompt to PostgreSQL Database!")

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
            item = {"inferenceId": inference[0], "userId": inference[1], "chatId": inference[2], "initialPrompt": inference[3], "finalPrompt": inference[4], "response_raw": inference[5], "createdAt": inference[6], "context_ids": inference[7], "context_scores": inference[8], "response_html": response_html}
            inferences.append(item)
        
    
    except Exception as e:
        loggingConfig["loggingObject"].exception("Error when Listing Inferences to PostgreSQL Database!")
        print(e)
        raise Exception("Error when Listing Inferences to PostgreSQL Database!")

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
            item = {"id": doc[0], "name": doc[1], "type": doc[2], "createdAt": doc[4]}
            docs.append(item)
    
    except Exception as e:
        loggingConfig["loggingObject"].exception(f"Error when retrieving Document {documentId} from PostgreSQL Database!")
        print(e)
        raise Exception(f"Error when retrieving Document {documentId} from PostgreSQL Database!")

    finally:
        cursor.close()
        conn.close()

    # Populate the docs Array if no docs matches
    if not docs:
        item = {"id": "NOT_FOUND", "name": "NOT_FOUND", "type": "NOT_FOUND", "date": "NOT_FOUND"}
        docs.append(item) 

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
        raise Exception(f"Error when deleting Document {documentId} PostgreSQL Database!")

    finally:
        cursor.close()
        conn.close()
        
    loggingConfig["loggingObject"].info(f"Document {documentId} deleted Successfully from PostgreSQL Database.")

    return True

'''
Inference V2 - Utils
'''
def savePrompt_v2(userId, chatId, initialPrompt, augmentedPrompt, finalPrompt, response, psqlConnectionConfig, loggingConfig, inferenceId, context_ids, context_scores):
    currentDate = datetime.now()

    try:
        loggingConfig["loggingObject"].info("[V2] Saving Prompt Started.")
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = f"INSERT INTO inferencesV2(inferenceId, initialPrompt, finalPrompt1, finalPrompt2, createdAt, response, userId, chatId, context_ids, context_scores) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

        cursor.execute(sql_query, (inferenceId, initialPrompt, augmentedPrompt, finalPrompt, currentDate, response, userId, chatId, context_ids, context_scores))

        conn.commit()
    
    except Exception as e:
        loggingConfig["loggingObject"].exception("Error when Saving Prompt to PostgreSQL Database!")
        print(e)
        raise Exception("Error when Saving Prompt to PostgreSQL Database!")

    finally:
        cursor.close()
        conn.close()
        
    loggingConfig["loggingObject"].info("Saving Prompt Finished.")
    return True


'''
Chat V1 - Utils
'''
def saveInference_ChatV1_to_postgresql(userId, chatId, initialPrompt, finalPrompt, response, psqlConnectionConfig, loggingConfig, inferenceId, context_ids, context_scores):
    currentDate = datetime.now()

    try:
        loggingConfig["loggingObject"].info(f"[Chat V1 | {inferenceId}] Adding Inference to PostgreSQL Database.")
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = f"INSERT INTO inferences_ChatV1(inferenceId, initialPrompt, finalPrompt, createdAt, response, userId, chatId, context_ids, context_scores) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"

        cursor.execute(sql_query, (inferenceId, initialPrompt, finalPrompt, currentDate, response, userId, chatId, context_ids, context_scores))

        conn.commit()
    
    except Exception as e:
        loggingConfig["loggingObject"].exception(f"[Chat V1 | {inferenceId}] Error when adding Inference to PostgreSQL Database.")
        print(e)
        raise Exception(f"[Chat V1 | {inferenceId}] Error when adding Inference to PostgreSQL Database.")

    finally:
        cursor.close()
        conn.close()
        
    loggingConfig["loggingObject"].info(f"[Chat V1 | {inferenceId}] Addedd Inference to PostgreSQL Database successfully!")
    return True

def getInference_ChatV1_from_postgresql(chatId, psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Chat V1 | Chat: {chatId}] Listing Inferences started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM inferences_ChatV1 WHERE chatId = %s;"

        cursor.execute(sql_query, (chatId,))

        conn.commit()

        data = cursor.fetchall()

        """
        print(data)
        i = 0
        for result in data:
            while(i < len(result)):
                print(f"[{i}] : {result[i]}")
                i += 1
        """
        
        inferences = []
        
        for inference in data:
            response_raw = inference[5]
            response_html = markdown.markdown(response_raw)


            # Get Documents Data
            context_ids = inference[7]
            docs = []
            for documentId in context_ids:
                doc = getDocumentById(documentId, psqlConnectionConfig, loggingConfig)
                docs.append(doc)

            item = {"inferenceId": inference[0], "userId": inference[1], "chatId": inference[2], "initialPrompt": inference[3], "finalPrompt": inference[4], "response_raw": inference[5], "createdAt": inference[6], "context_ids": inference[7], "context_scores": inference[8], "response_html": response_html, "docs": docs}
            inferences.append(item)
        

        loggingConfig["loggingObject"].info(f"[Chat V1 | Chat: {chatId}] Listing Inferences success!")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat V1 | Chat: {chatId}] Listing Inferences failled!")
        print(e)
        raise Exception(f"[Chat V1 | Chat: {chatId}] Listing Inferences failled!")


    finally:
        cursor.close()
        conn.close()

    
    return inferences

'''
Chat Management Utils
'''
def getChatsbyUserId(userId, psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Chat Management V1 | User: {userId}] Listing Chats started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM chats WHERE userId = %s;"

        cursor.execute(sql_query, (userId,))

        conn.commit()

        data = cursor.fetchall()

    
        '''
        print(data)
        i = 0
        for result in data:
            while(i < len(result)):
                print(f"[{i}] : {result[i]}")
                i += 1
        '''
        
        chats = []
        
        for chat in data:
            item = {"chatId": chat[0], "userId": chat[1], "name": chat[2], "createdAt": chat[3]}
            chats.append(item)
        

        loggingConfig["loggingObject"].info(f"[Chat Management V1 | User: {userId}] Listing Chats successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat Management V1 | User: {userId}] Listing Chats failed.")
        print(e)
        raise Exception(f"[Chat Management V1 | User: {userId}] Listing Chats failed.")


    finally:
        cursor.close()
        conn.close()

    
    return chats

def createChat(userId, psqlConnectionConfig, loggingConfig, chatId, name):
    loggingConfig["loggingObject"].info(f"[Chat Management V1 | User: {userId}] Creating Chat {chatId} started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        currentDate = datetime.now()

        sql_query = f"INSERT INTO chats(chatId, userId, name, createdAt) VALUES (%s, %s, %s, %s);"

        cursor.execute(sql_query, (chatId, userId, name, currentDate))

        conn.commit()

        loggingConfig["loggingObject"].info(f"[Chat Management V1 | User: {userId}] Creating Chat {chatId} successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat Management V1 | User: {userId}] Creating Chat {chatId} failed.")
        print(e)
        raise Exception(f"[Chat Management V1 | User: {userId}] Creating Chat {chatId} failed.")

    finally:
        cursor.close()
        conn.close()

    
    return True


'''
Customer Management Utils
'''
def createCustomer(customerId, psqlConnectionConfig, loggingConfig, name):
    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Creating Customer {customerId} started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        currentDate = datetime.now()

        sql_query = f"INSERT INTO customers(customerId, name, createdAt, updatedAt) VALUES (%s, %s, %s, %s);"

        cursor.execute(sql_query, (customerId, name, currentDate, currentDate))

        conn.commit()

        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Creating Customer {customerId} successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Creating Customer {customerId} failed.")
        print(e)
        raise Exception(f"[Customer Management V1 | Customer: {customerId}] Creating Customer {customerId} failed.")

    finally:
        cursor.close()
        conn.close()

    
    return True

def updateCustomer(customerId, psqlConnectionConfig, loggingConfig, field, value):
    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Updating Customer {customerId} started.")
    
    ALLOWED_FIELDS = {
        "name",
        "profile",
        "products",
        "contacts"
    }

    try:
        
        if field not in ALLOWED_FIELDS:
            raise ValueError("Invalid value for 'field' name")
    

        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        currentDate = datetime.now()

        sql_query = f"UPDATE customers SET {field} = %s, updatedAt = %s WHERE customerId = %s;"

        cursor.execute(sql_query, (value, currentDate, customerId))

        conn.commit()

        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Updating Customer {customerId} successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Updating Customer {customerId} failed.")
        print(e)
        raise Exception(f"[Customer Management V1 | Customer: {customerId}] Updating Customer {customerId} failed.")

    finally:
        cursor.close()
        conn.close()

    
    return True


def getCustomers(psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Customer Management V1] Listing Customers started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM customers;"

        cursor.execute(sql_query)

        conn.commit()

        data = cursor.fetchall()
        
        
        # expandPsqlOutput(data)
        
        customers = []
        
        for customer in data:
            item = {"customerId": customer[0], "name": customer[1], "profile": customer[2], "products": customer[3], "contacts": customer[4], "createdAt": customer[5], "updatedAt": customer[6]}
            customers.append(item)       

        loggingConfig["loggingObject"].info(f"[Customer Management V1] Listing Customers successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Customer Management V1] Listing Customers failed.")
        print(e)

    finally:
        cursor.close()
        conn.close()

    
    return customers

def getCustomerById(customerId, psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Customer Management V1] Listing Customers started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM customers WHERE customerId = %s;"

        cursor.execute(sql_query, (customerId,))

        conn.commit()

        data = cursor.fetchall()
        
        
        # expandPsqlOutput(data)
        
        customers = []
        
        for customer in data:
            item = {"customerId": customer[0], "name": customer[1], "profile": customer[2], "products": customer[3], "contacts": customer[4], "createdAt": customer[5], "updatedAt": customer[6]}
            customers.append(item)       

        loggingConfig["loggingObject"].info(f"[Customer Management V1] Listing Customers successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Customer Management V1] Listing Customers failed.")
        print(e)
        raise Exception(f"[Customer Management V1] Listing Customers failed.")

    finally:
        cursor.close()
        conn.close()

    
    return customers


def getCustomerById(customerId, psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Getting Customer data started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM customers WHERE customerId = %s;"

        cursor.execute(sql_query, (customerId,))

        conn.commit()

        data = cursor.fetchall()

        # expandPsqlOutput(data)

        customers = []
        
        for customer in data:
            item = {"customerId": customer[0], "name": customer[1], "profile": customer[2], "products": customer[3], "contacts": customer[4], "createdAt": customer[5], "updatedAt": customer[6]}
            customers.append(item)
        
        if not customers:
            raise Exception(f"[Customer Management V1 | Customer: {customerId}] Customer doesn't exist.")

        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Getting Customer data successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Getting Customer data failed.")
        print(e)
        raise Exception(f"[Customer Management V1 | Customer: {customerId}] Getting Customer data failed.")


    finally:
        cursor.close()
        conn.close()

    
    return customers[0]


def getDocumentsByCustomerById(customerId, psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Getting Documents for Customer started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM documents WHERE purpose = %s;"

        cursor.execute(sql_query, (customerId,))

        conn.commit()

        data = cursor.fetchall()

        # expandPsqlOutput(data)

        docs = []
        
        for doc in data:
            item = {"id": doc[0], "name": doc[1], "type": doc[2], "purpose": doc[3], "createdAt": doc[4], "updatedAt": doc[5]}
            docs.append(item)
        
        if not docs:
            raise Exception(f"[Customer Management V1 | Customer: {customerId}] Customer doesn't exist.")

        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Getting Customer data successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Customer Management V1 | Customer: {customerId}] Getting Customer data failed.")
        print(e)
        raise Exception(f"[Customer Management V1 | Customer: {customerId}] Getting Customer data failed.")


    finally:
        cursor.close()
        conn.close()

    
    return docs


def expandPsqlOutput(data):
    i = 0
    for result in data:
            while(i < len(result)):
                print(f"[{i}] : {result[i]}")
                i += 1
    
    return True


def getRegulationDocs(psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Regulation Management V1] Getting Documents for REGULATION(s) started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM documents WHERE purpose = 'REGULATION';"

        cursor.execute(sql_query)

        conn.commit()

        data = cursor.fetchall()

        # expandPsqlOutput(data)

        docs = []
        
        for doc in data:
            item = {"id": doc[0], "name": doc[1], "type": doc[2], "purpose": doc[3], "createdAt": doc[4], "updatedAt": doc[5]}
            docs.append(item)
        
        loggingConfig["loggingObject"].info(f"[Regulation Management V1] Getting REGULATION(s) data successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Regulation Management V1] Getting REGULATION(s) data failed.")
        print(e)
        raise Exception(f"[Regulation Management V1] Getting REGULATION(s) data failed.")


    finally:
        cursor.close()
        conn.close()

    
    return docs

def getAllCustomerDocs(psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Regulation Management V1] Getting Documents for REGULATION(s) started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM documents WHERE purpose != 'REGULATION' AND purpose != 'PRODUCT';"

        cursor.execute(sql_query)

        conn.commit()

        data = cursor.fetchall()

        # expandPsqlOutput(data)

        docs = []
        
        for doc in data:
            item = {"id": doc[0], "name": doc[1], "type": doc[2], "purpose": doc[3], "createdAt": doc[4], "updatedAt": doc[5]}
            docs.append(item)
        
        loggingConfig["loggingObject"].info(f"[Regulation Management V1] Getting REGULATION(s) data successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Regulation Management V1] Getting REGULATION(s) data failed.")
        print(e)
        raise Exception(f"[Regulation Management V1] Getting REGULATION(s) data failed.")


    finally:
        cursor.close()
        conn.close()

    
    return docs


def getAllProductDocs(psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Product Knowledge Management V1] Getting Documents for PRODUCTS(s) started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM documents WHERE purpose = 'PRODUCT';"


        cursor.execute(sql_query)

        conn.commit()

        data = cursor.fetchall()

        # expandPsqlOutput(data)

        docs = []
        
        for doc in data:
            item = {"id": doc[0], "name": doc[1], "type": doc[2], "purpose": doc[3], "createdAt": doc[4], "updatedAt": doc[5]}
            docs.append(item)
        
        
        loggingConfig["loggingObject"].info(f"[Product Knowledge Management V1] Getting PRODUCTS(s) data successful.")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Product Knowledge Management V1] Getting PRODUCTS(s) data failed.")
        print(e)
        raise Exception(f"[Product Knowledge Management V1] Getting PRODUCTS(s) data failed.")


    finally:
        cursor.close()
        conn.close()

    
    return docs

'''
For Chat V2
'''
def getRecentLLMInferencesByChatId_chat_v2(chatId, amount,  psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Chat V2 | Chat: {chatId}] Listing Recent {amount} Inferences started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM inferences_ChatV2 WHERE chatId = %s ORDER BY createdAt DESC LIMIT %s;"

        cursor.execute(sql_query, (chatId, amount))

        conn.commit()

        data = cursor.fetchall()
        
        inferences = []
        
        for inference in data:
            response_raw = inference[5]
            response_html = markdown.markdown(response_raw)


            # Get Documents Data
            context_ids = inference[7]
            docs = []
            for documentId in context_ids:
                doc = getDocumentById(documentId, psqlConnectionConfig, loggingConfig)
                docs.append(doc)

            item = {"inferenceId": inference[0], "userId": inference[1], "chatId": inference[2], "initialPrompt": inference[3], "finalPrompt": inference[4], "response_raw": inference[5], "createdAt": inference[6], "context_ids": inference[7], "context_scores": inference[8], "response_html": response_html, "docs": docs}
            inferences.append(item)
        

        loggingConfig["loggingObject"].info(f"[Chat V2 | Chat: {chatId}] Listing Recent {amount} Inferences success!")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat V2 | Chat: {chatId}] Listing Recent {amount} Inferences failled!")
        print(e)
        raise Exception(f"[Chat V2 | Chat: {chatId}] Listing Recent {amount} Inferences failled!")


    finally:
        cursor.close()
        conn.close()

    
    return inferences


def saveInference_ChatV2_to_postgresql(userId, chatId, initialPrompt, finalPrompt, response, psqlConnectionConfig, loggingConfig, inferenceId, context_ids, context_scores):
    currentDate = datetime.now()

    try:
        loggingConfig["loggingObject"].info(f"[Chat V2 | {inferenceId}] Adding Inference to PostgreSQL Database.")
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = f"INSERT INTO inferences_ChatV2(inferenceId, initialPrompt, finalPrompt, createdAt, response, userId, chatId, context_ids, context_scores) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"

        cursor.execute(sql_query, (inferenceId, initialPrompt, finalPrompt, currentDate, response, userId, chatId, context_ids, context_scores))

        conn.commit()
    
    except Exception as e:
        loggingConfig["loggingObject"].exception(f"[Chat V2 | {inferenceId}] Error when adding Inference to PostgreSQL Database.")
        print(e)
        raise Exception(f"[Chat V2 | {inferenceId}] Error when adding Inference to PostgreSQL Database.")

    finally:
        cursor.close()
        conn.close()
        
    loggingConfig["loggingObject"].info(f"[Chat V2 | {inferenceId}] Addedd Inference to PostgreSQL Database successfully!")
    return True

def getInference_ChatV2_from_postgresql(chatId, psqlConnectionConfig, loggingConfig):
    loggingConfig["loggingObject"].info(f"[Chat V2 | Chat: {chatId}] Listing Inferences started.")
    try:
        conn = get_db_connection(psqlConnectionConfig)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM inferences_ChatV2 WHERE chatId = %s;"

        cursor.execute(sql_query, (chatId,))

        conn.commit()

        data = cursor.fetchall()
        
        inferences = []
        
        for inference in data:
            response_raw = inference[5]
            response_html = markdown.markdown(response_raw)


            # Get Documents Data
            context_ids = inference[7]
            docs = []
            for documentId in context_ids:
                doc = getDocumentById(documentId, psqlConnectionConfig, loggingConfig)
                docs.append(doc)

            item = {"inferenceId": inference[0], "userId": inference[1], "chatId": inference[2], "initialPrompt": inference[3], "finalPrompt": inference[4], "response_raw": inference[5], "createdAt": inference[6], "context_ids": inference[7], "context_scores": inference[8], "response_html": response_html, "docs": docs}
            inferences.append(item)
        

        loggingConfig["loggingObject"].info(f"[Chat V2 | Chat: {chatId}] Listing Inferences success!")
    
    except Exception as e:
        loggingConfig["loggingObject"].info(f"[Chat V2 | Chat: {chatId}] Listing Inferences failled!")
        print(e)
        raise Exception(f"[Chat V2 | Chat: {chatId}] Listing Inferences failled!")


    finally:
        cursor.close()
        conn.close()

    
    return inferences
