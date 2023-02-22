'''Add a user to the database'''
from register_face import register_face
from recognize_face import recognize_face
from users import Users

# Create an instance of the Users class
users = Users()

# Register a user
user, face_location = register_face("path/to/image.jpg")
users.add_user(user)

# Recognize a face
name = recognize_face("path/to/image.jpg", users.get_users())
print(name)
