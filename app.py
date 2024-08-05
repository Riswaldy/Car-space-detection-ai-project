from flask import Flask, render_template, request, Response, jsonify
import cv2
import pickle
import cvzone
import numpy as np
import mysql.connector
from math import ceil

app = Flask(__name__)

def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="",
        database="parking_project"
    )
    return conn

@app.route('/list_parking')
def list_parking():
    search_query = request.args.get('search', '')
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Count total results for pagination
    count_query = """
        SELECT COUNT(*) FROM parking_data
        WHERE parking_number LIKE %s OR status LIKE %s
    """
    cursor.execute(count_query, (f'%{search_query}%', f'%{search_query}%'))
    total_results = cursor.fetchone()['COUNT(*)']
    total_pages = ceil(total_results / per_page)
    
    # Fetch paginated and filtered results
    start = (page - 1) * per_page
    select_query = """
        SELECT * FROM parking_data
        WHERE parking_number LIKE %s OR status LIKE %s
        LIMIT %s OFFSET %s
    """
    cursor.execute(select_query, (f'%{search_query}%', f'%{search_query}%', per_page, start))
    parking_data = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template('list_parking.html', parking_data=parking_data, page=page, total_pages=total_pages, search=search_query)

@app.route('/note')
def note():
    search_query = request.args.get('search', '')
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Count total results for pagination
    count_query = """
        SELECT COUNT(*) FROM parking_data
        WHERE parking_number LIKE %s OR status LIKE %s
    """
    cursor.execute(count_query, (f'%{search_query}%', f'%{search_query}%'))
    total_results = cursor.fetchone()['COUNT(*)']
    total_pages = ceil(total_results / per_page)
    
    # Fetch paginated and filtered results
    start = (page - 1) * per_page
    select_query = """
        SELECT * FROM parking_data
        WHERE parking_number LIKE %s OR status LIKE %s
        LIMIT %s OFFSET %s
    """
    cursor.execute(select_query, (f'%{search_query}%', f'%{search_query}%', per_page, start))
    parking_data = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template('note.html', parking_data=parking_data, page=page, total_pages=total_pages, search=search_query)

@app.route('/update_note', methods=['POST'])
def update_note():
    updated_data = request.json
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for row in updated_data:
        update_query = """
            UPDATE parking_data
            SET status = %s
            WHERE parking_number = %s
        """
        cursor.execute(update_query, (row['status'], row['parking_number']))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({'message': 'Data updated successfully!'}), 200

# Load parking position data
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

def checkParkingSpace(imgPro, img):
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y+height, x:x+width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cvzone.putTextRect(img, str(count), (x, y+height-3), scale=1, thickness=2, offset=0, colorR=color)
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), color, thickness)
    
    cvzone.putTextRect(img, f'Free: {spaceCounter} / {len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))

def gen_frames():
    cap = cv2.VideoCapture('carPark.mp4')
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, img = cap.read()
        if not success:
            break
        else:
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
            
            checkParkingSpace(imgDilate, img)
            
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/view_camera')
def view_camera():
    return render_template('view_camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
