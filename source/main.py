import os
import sqlite3
from typing import List

import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()

# Connect to the database
conn = sqlite3.connect("users.db")
c = conn.cursor()

# Create the users table if it doesn't exist
c.execute(
    """CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, face_encoding BLOB)"""
)
conn.commit()


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h1>Welcome to the Face Recognition Attendance Management System</h1>
        </body>
    </html>
    """


@app.post("/register")
async def register(name: str, email: str, file: UploadFile = File(...)):
    # Read the image file from the request
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect the face in the image
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) != 1:
        return {"error": "Please upload an image with exactly one face."}

    # Encode the face and add the user to the database
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    encoded_face = face_encoding.tobytes()
    c.execute(
        "INSERT INTO users (name, email, face_encoding) VALUES (?, ?, ?)",
        (name, email, encoded_face),
    )
    conn.commit()

    return {"message": f"User {name} has been registered."}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    # Read the image file from the request
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect the face in the image
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) != 1:
        return {"error": "Please upload an image with exactly one face."}

    # Encode the face and compare it to the faces in the database
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    encoded_face = face_encoding.tobytes()
    c.execute("SELECT name, email FROM users")
    rows = c.fetchall()
    return next(
        (
            {"name": row[0], "email": row[1]}
            for row in rows
            if face_recognition.compare_faces(
                [np.frombuffer(row[2], dtype=np.float64)], face_encoding
            )
        ),
        {"error": "No matching user found."},
    )


@app.get("/users")
def get_users() -> List[dict]:
    # Retrieve all users from the database
    c.execute("SELECT name, email FROM users")
    rows = c.fetchall()
    return [{"name": row[0], "email": row[1]} for row in rows]


@app.put("/users/{name}")
async def update_user(name: str, email: str, file: UploadFile = File(None)):
    # Update the email for the specified user
    c.execute("UPDATE users SET email=? WHERE name=?", (email, name))
    conn.commit()

    # If
