# import tensorflow as tf

# modelPath = './model0326_crop.h5'
# tflitePath = './model0326_crop_Lite.tflite'
# model = tf.keras.models.load_model(modelPath)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflmodel = converter.convert()
# file = open(tflitePath, 'wb' ) 
# file.write( tflmodel )

import tensorflow as tf

## TFLite Conversion
# Before conversion, fix the model input size
modelPath = './model0326_crop.h5'
tflitePath = './model0326_crop_Lite.tflite'
model = tf.saved_model.load(modelPath)
model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[0].set_shape((2, 1, 128, 512, 1))
tf.saved_model.save(model, tflitePath, signatures=model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY])
# Convert
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=tflitePath, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()