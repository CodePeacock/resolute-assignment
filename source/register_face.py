import face_recognition

def register_face():
    known_image = face_recognition.load_image_file("known_person.jpg")
    known_face_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_image = face_recognition.load_image_file("unknown_person.jpg")
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_face_encoding in unknown_face_encodings:
    # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(
        [known_face_encoding], unknown_face_encoding
    )

        if any(matches):
            print("This is a picture of the known person!")
        else:
            print("This is not a picture of the known person.")

register_face()
