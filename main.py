import tensorflow as tf
import cv2 as cv
import utils
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from fastapi import File
from fastapi import UploadFile


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#post function
@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    image = cv.imread(f"files/{file.filename}")

    # Loading Model
    sess = tf.compat.v1.Session()
    model = tf.saved_model.loader.load(sess ,tags = ['serve'], export_dir = 'model_pb')

    # Get Predicted Text
    resized_image = tf.image.resize_image_with_pad(image, 64, 1024).eval(session = sess)
    img_gray = cv.cvtColor(resized_image, cv.COLOR_RGB2GRAY).reshape(64,1024,1)

    output = sess.run('Dense-Decoded/SparseToDense:0', 
            feed_dict = {
                'Deep-CNN/Placeholder:0':img_gray
            })
    output_text = utils.dense_to_text(output[0])

    print(output_text)

    return {
        "output_text":output_text
    }