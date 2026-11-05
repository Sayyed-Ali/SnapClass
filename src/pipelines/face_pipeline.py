import numpy as np
import dlib
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students

@st.cache_resource # so that the models are loaded only once as they are quite heavy
def load_dlib_models():
    detector = dlib.get_frontal_face_detector() # gives the bonding box for face scan

    sp = dlib.shape_predictor(       # to detect the face landmarks
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)   # 1 to scann image only once (dont do rotation and oter things to avoid computer resources usage) cacnn try with oyther values as well

    encodings = []
    
    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1) # 128 embedding

        encodings.append(np.array(face_descriptor))
    
    return encodings


# train a small model on all the students in our database (SVC)
@st.cache_resource
def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        None
    
    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))
    
    if len(X)==0:
        return None
    
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')

    try:
        clf.fit(X, y)
    except ValueError:
        pass

        return {'clf': clf, 'X': X, 'y': y}



def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)


def predict_attendence(class_image_np):
    encodings = get_face_embeddings(class_image_np)     # will give all embedding from img

    detected_student = {}

    model_data = get_trained_model()

    if not model_data:
        return detected_student, [], len(encodings)
    
    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))

    for encoding in encodings:
        if len(all_students)>=2:
            predicted_id = int(clf.predict([encoding])[0])
        else:
            predicted_id = int(all_students[0])

        student_embedding = X_train[y_train.index(predicted_id)]

        best_match_score = np.linalg.norm(student_embedding - encoding)

        resemblance_threshold = 0.6
        if best_match_score < resemblance_threshold:
            detected_student[predicted_id] = True

    return detected_student, all_students, len(encodings) # detected student, all students on which we have worked and number of students





