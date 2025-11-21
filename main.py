import face_recognition
import cv2
import os

# -------------------------
# 1. Load Haar Cascades (Eyes + Smile)
# -------------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# -------------------------
# 2. Load Known Faces (Face Recognition)
# -------------------------
known_face_encodings = []
known_face_names = []

path = "known_faces"   # Folder structure: known_faces/Anu/*.jpg
for name in os.listdir(path):
    person_folder = os.path.join(path, name)

    for image_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image_name)

        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)

# -------------------------
# 3. Start Webcam
# -------------------------
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Face Recognition Detection ---
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    # Loop through each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Draw main Face Box
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Put Face Name
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # -------------------------
        # Haar Cascade Face ROI for Eyes + Smile
        # -------------------------
        face_roi_gray = gray[top:bottom, left:right]
        face_roi_color = frame[top:bottom, left:right]

        # Detect Eyes
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.2, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Detect Smile
        smiles = smile_cascade.detectMultiScale(face_roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)

    # Show Window
    cv2.imshow("Face Recognition + Eyes + Smile Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()