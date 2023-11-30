from pathlib import Path
import os
from werkzeug.utils import secure_filename

from flask import Flask, request, flash, redirect

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'test'

# This sets up CORS to allow browsers to actually get this data?
# The default is pretty permissive but I think that's fine?
# Look into this further at some point.
# CORS(app)

@app.route('/api/yaml', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file:
            filename = secure_filename(file.filename)

            upload_folder = Path(app.config["UPLOAD_FOLDER"])
            upload_folder.mkdir(parents=True, exist_ok=True)

            filepath = upload_folder / filename
            file.save(filepath)

            return {}, 200
        else:
            return {}, 402


if __name__ == "__main__":
    app.run()
