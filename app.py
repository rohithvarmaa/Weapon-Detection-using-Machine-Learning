from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)

# Define YOLO configurations
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]

# Get YOLO layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set up an uploads folder for temporary storage
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def process_file(file_path):
    cap = cv2.VideoCapture(file_path)

    weapon_detected = False  # Flag to indicate if a weapon is detected

    while True:
        _, img = cap.read()
        if img is None:
            break

        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    
                    w = int(detection[2] * width) if detection[2] * width < float('inf') else 0


                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            print("Weapon detected in frame")
            weapon_detected = True  # Set the flag to True if a weapon is detected



        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        # Display the frame
        cv2.imshow("Frame", img)

        # Save the processed image temporarily
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        cv2.imwrite(output_path, img)

        # Break the loop if 'Esc' key is pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return output_path, weapon_detected  # Return the output path and the weapon detection flag



# Route for the index page
@app.route('/')
def index():
    return render_template('index.html', result_image=None)

# Route to handle file upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the 'file' key is in the request.files dictionary
    if 'file' not in request.files:
        return render_template('index.html', result_image=None, result="No file selected!")

    uploaded_file = request.files['file']

    # Check if the file has a valid filename
    if uploaded_file.filename == '':
        return render_template('index.html', result_image=None, result="No file selected!")

    # Save the uploaded file temporarily
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(file_path)

    # Process the uploaded file
    output_image_path, weapon_detected = process_file(file_path)

    # Provide a result message based on weapon detection status
    if weapon_detected:
        result = "Weapon detected!"
    else:
        result = "No weapon detected."

    return render_template('index.html', result_image=output_image_path, result=result)

# Route to serve the processed image
@app.route('/result_image')
def get_result_image():
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
    return send_file(result_image_path, mimetype='image/jpg', download_name='output.jpg')

if __name__ == '__main__':
    app.run(port=5014, debug=True)
