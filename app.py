from flask import Flask,render_template

import os
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET'])
def upload_file_page():
    return render_template("upload.html")



@app.route('/v1/upload', methods=['POST'])
def upload_file():
    
    
    file = request.files['file']

    print(file)

    filename = secure_filename(file.filename)

    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return {
        "status": "success",
        "message": "Upload Success!"
    }

app.run(host="0.0.0.0", port=5000)