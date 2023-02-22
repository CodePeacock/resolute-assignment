'''Recognize a face in an image using a trained model'''
from typing import Dict

import face_recognition


def recognize_face(image_path: str, users: Dict) -> str:
    '''Recognize a face in an image using a trained model'''
    image = face_recognition.load_image_file(image_path)

    # Find the face location and encoding
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Iterate through each user to find a match
    for user in users.values():
        user_encoding = np.load(user["encoding_file_path"])

        # Compare face encodings
        matches = face_recognition.compare_faces([user_encoding], face_encodings[0])
        if matches[0]:
            return user["name"]

    return "Unknown"
