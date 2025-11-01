import psycopg2
from datetime import datetime


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