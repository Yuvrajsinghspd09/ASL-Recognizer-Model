import http.server
from http import HTTPStatus
import json
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model

class MyHttpRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(post_data)
        image_data = data['image']
        predicted_letter = predict(image_data)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "application/json")
        
        # Handle CORS: Allow requests from any origin
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        
        self.end_headers()
        self.wfile.write(json.dumps({"predicted_letter": predicted_letter}).encode())

    # Handle OPTIONS request for CORS preflight
    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "application/json")
        
        # Handle CORS: Allow requests from any origin
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        
        self.end_headers()

def run(server_class=http.server.HTTPServer, handler_class=MyHttpRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server at port {port}")
    httpd.serve_forever()

def preprocess_image(image_data):
    # Convert base64 image data to OpenCV format
    img_bytes = BytesIO(image_data.encode('utf-8'))
    img_pil = Image.open(img_bytes)
    img_np = np.array(img_pil)
    
    # Perform preprocessing steps
    img_resized = cv2.resize(img_np, (200, 200))
    img_normalized = img_resized.astype('float32') / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=-1)
    img_final = np.expand_dims(img_expanded, axis=0)
    
    return img_final

def predict(image_data):
    # Preprocess the received image_data
    processed_image = preprocess_image(image_data)
    
    # Load your trained model
    # Replace 'my_model.keras' with the path to your .keras model
    model = load_model('my_model.keras')
    
    # Perform prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return chr(predicted_class + ord('A'))

if __name__ == "__main__":
    run()

