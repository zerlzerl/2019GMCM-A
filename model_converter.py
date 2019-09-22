import time

import tensorflow as tf
from tensorflow.python import keras

h5_model_path = "./model/my_model.h5"
tf.keras.backend.set_learning_phase(0)
model = keras.models.load_model(h5_model_path)
export_path = './model/tf/' + str(int(time.time()))
print(model.inputs)
print(model.outputs)
#
with tf.keras.backend.get_session() as sess:
    # model_input = tf.placeholder(tf.float32, [None, 44])
    # model_output = tf.placeholder(tf.float32, [None])
    model_input = tf.placeholder(tf.float32, [None, 52])
    model_output = model(model_input)
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'myInput': model_input},
        outputs={'myOutput': model_output})


