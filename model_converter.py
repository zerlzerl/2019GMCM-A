import time

import tensorflow as tf
import keras

h5_model_path = "./model/0922/weights-improvement-99-84.54.h5"
tf.keras.backend.set_learning_phase(0)
model = keras.models.load_model(h5_model_path)
export_path = './model/tf/1200w_origin_99epoch'
print(model.inputs)
print(model.outputs)
#
with tf.keras.backend.get_session() as sess:
    # model_input = tf.placeholder(tf.float32, [None, 44])
    # model_output = tf.placeholder(tf.float32, [None])
    model_input = tf.placeholder(tf.float32, [None, 27])
    model_output = model(model_input)
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'myInput': model_input},
        outputs={'myOutput': model_output})


