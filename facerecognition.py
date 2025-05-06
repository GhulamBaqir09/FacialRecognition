import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import numpy as np

model_name = 'Facenet'


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

reference_images = {
    "Baqir": r"C:\Users\Admin\OneDrive\Desktop\hh\baqirrr.jpg",
    "Bisma": r"C:\Users\Admin\OneDrive\Desktop\hh\bisma.jpeg",
    "Imran Khan": r"C:\Users\Admin\OneDrive\Desktop\hh\Imrankhan.jfif"
}

reference_embeddings = {}

for name, img_path in reference_images.items():
    try:
        embedding_info = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=True)
        reference_embeddings[name] = embedding_info[0]['embedding']
        print(f"Embedding extracted for {name}")
    except Exception as e:
        print(f"Failed to process {name}: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_image = rgb_frame[y:y+h, x:x+w]
        try:
           
            resized_face = cv2.resize(face_image, (160, 160)) 

            face_embedding = DeepFace.represent(img_path=resized_face, model_name=model_name, enforce_detection=False)[0]['embedding']

            matched_label = "Not Matched"
            min_distance = 1.0
            for name, ref_embedding in reference_embeddings.items():
                dist = cosine(face_embedding, ref_embedding)
                print(f"Distance to {name}: {dist:.4f}")
                if dist < min_distance and dist < 0.75: 
                    min_distance = dist
                    matched_label = name

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, matched_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        except Exception as e:
            print(f"Embedding error: {e}")

    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
