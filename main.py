import cv2
import numpy as np
import tensorflow as tf

# Path to label map file
PATH_TO_LABELS = "cat_dog_labels.txt"

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="cat_dog_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input size
input_shape = input_details[0]['shape']
input_size = (input_shape[1], input_shape[2])

# Capture video from the built-in camera
cap = cv2.VideoCapture(0)

last_detected = None  # Track the last detected object
detection_pause_frames = 0  # Counter to reset last_detected
detection_pause_threshold = 300  # Number of frames to wait before resetting, adjust as needed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    frame_resized = cv2.resize(frame, input_size)

    # Convert frame to float32
    frame_resized = frame_resized.astype(np.float32)

    # Normalize the frame if required by your model
    input_data = frame_resized / 255.0

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)

    # Feed the frame to the model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve the output of the model
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Output data
    probability = output_data[0]  # Assuming single sigmoid output representing probability of the positive class

    # Determine class based on probability
    threshold = 0.98  # You can adjust the threshold as needed
    class_id = int(probability > threshold)
    score = probability if class_id == 1 else 1 - probability

    object_detected = score > threshold
    object_name = labels[class_id] if object_detected else "Unknown"

    # If no object is detected, increment the counter
    if not object_detected:
        detection_pause_frames += 1
    else:
        detection_pause_frames = 0  # Reset counter if an object is detected

    # Reset last_detected if the threshold is reached without detection
    if detection_pause_frames >= detection_pause_threshold:
        last_detected = None

    # Check if the detected object has changed or re-entered
    if object_detected and (object_name != last_detected):
        print(f"Detected {object_name} with confidence {score*100:.2f}%")
        last_detected = object_name  # Update the last detected object

    # Display the resulting frame
    cv2.imshow('Object Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
