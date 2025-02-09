from flask import Flask, send_from_directory, request, jsonify
import os
import subprocess


app = Flask(__name__, static_folder='web', static_url_path='')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return send_from_directory('web', 'surveillance.html')

@app.route('/about_us')
def about_us():
    return send_from_directory('web', 'Aboutus.html')

@app.route('/contact_us')
def contact_us():
    return send_from_directory('web', 'Contactus.html')

@app.route('/crowd_detection')
def crowd_detection():
    return send_from_directory('web', 'CrowdDetection.html')

@app.route('/fall_detection')
def fall_detection():
    return send_from_directory('web', 'FallDetection.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename}), 200
    
    return jsonify({'success': False, 'message': 'Failed to upload file'}), 500

@app.route('/detect_video/<filename>')
def detect_video(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'message': 'File not found'}), 404
    
    subprocess.Popen(["python", "run.py", video_path])

    return jsonify({'success': True, 'message': 'Detection started.'}), 200

@app.route('/upload_fall_video', methods=['POST'])
def upload_fall_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename}), 200
    
    return jsonify({'success': False, 'message': 'Failed to upload file'}), 500

@app.route('/detect_fall_video/<filename>')
def detect_fall_video(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'message': 'File not found'}), 404
    
    subprocess.Popen(["python", "fall.py", video_path])

    return jsonify({'success': True, 'message': 'Fall detection started.'}), 200

@app.route('/start_fall_webcam_detection')
def start_fall_webcam_detection():
    subprocess.Popen(["python", "fall.py", "webcam"], shell=True)
    return jsonify({'success': True, 'message': 'Fall detection webcam started.'}), 200

@app.route('/start_crowd_webcam_detection')
def start_crowd_webcam_detection():
    subprocess.Popen(["python", "run.py", "webcam"], shell=True)
    return jsonify({'success': True, 'message': 'Crowd detection webcam started.'}), 200

@app.route('/web/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join('web', 'css'), filename)

@app.route('/web/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join('web', 'js'), filename)

@app.route('/web/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(os.path.join('web', 'images'), filename)

if __name__ == '__main__':
    app.run(debug=True)