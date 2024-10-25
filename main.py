from flask import Flask, Response, jsonify, request
import cv2
import threading
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="P23WgD6oL94DxdKRk7lJ"
)

# model = YOLO('best.pt')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
results = None


def send_notification(device_id):
    url = f"http://localhost:3333/notifications/{device_id}"
    data = {"deviceId": device_id}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Notificação enviada para o dispositivo {device_id}")
        else:
            print(f"Erro ao enviar notificação: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Erro ao enviar notificação: {e}")

def process_frame(frame):
    global results
    results = CLIENT.infer(frame, model_id="fall-detection-ca3o8/4")
    # results = model(frame)
    # print(results)


def process_face(faces, frame):
    for (x, y, w, h) in faces:
        if w <= 0 or h <= 0:
            continue

        face_img = frame[y:y + h, x:x + w]

        if face_img is None or face_img.size == 0:
            continue

        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            label = f'Age: {age}'
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            return age
        except Exception as e:
            print(f"Erro ao processar a imagem do rosto: {e}")


@app.route('/camera_feed', methods=['POST'])
def camera_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Captura da câmera
        if not cap.isOpened():
            print("Erro ao acessar a câmera")
            return
       
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            frame = cv2.resize(frame, (320, 240))


            threading.Thread(target=process_frame, args=(frame,)).start()

            if results:
                predictions = results.get('predictions', [])
                for prediction in predictions:
                    if prediction['class'] == 'Fall-Detected' and prediction['confidence'] >= 0.9:
                        print("Queda detectada!")

                        
                        age = process_face(faces, frame)

                        #Mandar para id do dispositivo
                        '''
                        /notifications/{deviceId}

                        '''
                        if age == '(60-100)':
                            send_notification(receive_id)
                        

        #     _, buffer = cv2.imencode('.jpg', frame)
        #     frame_bytes = buffer.tobytes()
        #     yield (b'--frame\r\n'
        #            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # cap.release()

    threading.Thread(target=generate).start() 
    return jsonify({"message": "Monitoramento iniciado"}), 200

@app.route('/receive_id', methods=['POST'])
def receive_id():
    global received_id
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({'error': 'ID não fornecido'}), 400

    received_id = data['id']
    print(f"ID recebido: {received_id}")
    return jsonify({'message': 'ID recebido com sucesso', 'id': received_id}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3333)
