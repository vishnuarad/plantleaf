from flask import Flask, request, render_template
import pandas as pd
from app2 import predict_plant
from werkzeug.utils import secure_filename
import os
from supabase import create_client, Client

# Declare a Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'static\\uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}



# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def flexbox2():
    return render_template('welcomepage.html')
@app.route('/flexbox2')
def flexbox1():
    return render_template('flexbox2.html')
@app.route('/website')
def website():
    return render_template('website.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    '''# If the user does not select a file, the browser submits an empty part without a filename.
    if file.filename == '':
        return "No selected file"'''

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(file_path)
        file_path=file_path.replace("\\","/")
        prediction=predict_plant(file_path)
        print(file_path[6:])

        url= "https://usaixccmmgetklehwlss.supabase.co"
        key= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVzYWl4Y2NtbWdldGtsZWh3bHNzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTY5NTM4OTE0NiwiZXhwIjoyMDEwOTY1MTQ2fQ.-cvbhw1ehaihrlpZVbqZsOztciFhbA5QFAOERt6CHnc"
        supabase= create_client(url, key)
        temp=prediction
        response=supabase.table('medicinal_plants').select("desc").eq('plant',temp).execute().data
        descr=response[0]['desc'] if response!=[] else ""
        return render_template('website.html',output=prediction,path=file_path[6:],desc=descr)




if __name__ == '__main__':
    app.run(debug=True)
