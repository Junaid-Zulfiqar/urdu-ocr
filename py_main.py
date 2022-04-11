import tensorflow as tf
import cv2 as cv
import utils

# path = input('Input Image Path: ')
path = "test.jpg"
image = cv.imread(path)

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
