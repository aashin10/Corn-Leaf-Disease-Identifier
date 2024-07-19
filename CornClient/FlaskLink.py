from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import subprocess

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/welcomePage')
def welcome_page():
    return render_template('welcomePage.html')


@app.route('/nutrition')
def nutrition():
    return render_template('dashboard.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        image = request.files['image']
        image_path = os.path.join('assets/Unknown', image.filename)
        image.save(image_path)

        # Execute imageClassifierTest.py and capture its output
        output = subprocess.check_output(['python', 'imageClassifierTest.py'])
        output_str = output.decode('utf-8').strip()  # Convert bytes to string

        # Create a JSON response with the output
        response_data = {'output': output_str}
        return jsonify(response_data)


app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

UPLOAD_FOLDER = 'assets/user_dataset/Train'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files[]')
    if len(files) == 0:
        return jsonify({"error": "No files selected for uploading"}), 400

    for file in files:
        if file.filename == '':
            return jsonify({"error": "File with no filename"}), 400

        # Ensure the directory exists
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        file.save(file_path)

    return jsonify({
        "message": f"Successfully uploaded {len(files)} files to {UPLOAD_FOLDER}",
        "uploaded_files": [file.filename for file in files]
    })


@app.route('/splitDataset', methods=['POST'])
def split_dataset():
    # Implement your dataset splitting logic here
    subprocess.check_output(['python', 'UserDataSpliter.py'])
    return jsonify({"message": "Dataset split successfully"})


@app.route('/clientStart', methods=['POST'])
def client_start():
    subprocess.check_output(['python', 'client.py'])
    return jsonify({"message": "Client process has finished successfully!"})


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large"}), 413


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
