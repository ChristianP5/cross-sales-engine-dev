from datetime import datetime
import hashlib

def generateId(length):
    currentDate = datetime.now()
    id = hashlib.sha256(str(currentDate).encode()).hexdigest()[:length]

    return id
