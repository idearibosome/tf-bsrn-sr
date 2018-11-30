import argparse
import importlib
import os
import time

import numpy as np
import tensorflow as tf

import models

FLAGS = tf.flags.FLAGS

DEFAULT_MODEL = 'bsrn'

if __name__ == '__main__':
  tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
  tf.flags.DEFINE_string('cuda_device', '-1', 'CUDA device index to be used in the validation. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify this to employ GPUs.')

  tf.flags.DEFINE_string('restore_path', None, 'Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
  tf.flags.DEFINE_string('restore_target', None, 'Target of the restoration.')
  tf.flags.DEFINE_integer('restore_global_step', 0, 'Global step of the restored model. Some models may require to specify this.')
  
  tf.flags.DEFINE_string('scales', '4', 'Upscaling factors. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')

  tf.flags.DEFINE_string('input_path', 'LR', 'Base path of the input images.')
  tf.flags.DEFINE_string('output_path', 'SR', 'Base path of the upscaled images to be saved.')

  # parse model first and import them
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--model', default=DEFAULT_MODEL)
  pre_parsed = pre_parser.parse_known_args()[0]
  if (pre_parsed.model is not None):
    MODEL_MODULE = importlib.import_module('models.' + pre_parsed.model)


def main(unused_argv):
  # initialize
  FLAGS.bsrn_intermediate_outputs = True
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device
  scale_list = list(map(lambda x: int(x), FLAGS.scales.split(',')))
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # image reading session
  tf_image_read_graph = tf.Graph()
  with tf_image_read_graph.as_default():
    tf_image_read_path = tf.placeholder(tf.string, [])
    
    tf_image = tf.read_file(tf_image_read_path)
    tf_image = tf.image.decode_png(tf_image, channels=3, dtype=tf.uint8)
    
    tf_image_read = tf_image

    tf_image_read_init = tf.global_variables_initializer()
    tf_image_read_session = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
    ))
    tf_image_read_session.run(tf_image_read_init)

  # image saving session
  tf_image_save_graph = tf.Graph()
  with tf_image_save_graph.as_default():
    tf_image_save_path = tf.placeholder(tf.string, [])
    tf_image_save_image = tf.placeholder(tf.float32, [None, None, 3])
    
    tf_image = tf_image_save_image
    tf_image = tf.round(tf_image)
    tf_image = tf.clip_by_value(tf_image, 0, 255)
    tf_image = tf.cast(tf_image, tf.uint8)
    
    tf_image_png = tf.image.encode_png(tf_image)
    tf_image_save_op = tf.write_file(tf_image_save_path, tf_image_png)

    tf_image_save_init = tf.global_variables_initializer()
    tf_image_save_session = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
    ))
    tf_image_save_session.run(tf_image_save_init)

  # model
  model = MODEL_MODULE.create_model()
  model.prepare(is_training=False, global_step=FLAGS.restore_global_step)

  # model > restore
  model.restore(ckpt_path=FLAGS.restore_path, target=FLAGS.restore_target)
  tf.logging.info('restored the model')
  
  # get image path list
  image_list = [f for f in os.listdir(FLAGS.input_path) if f.lower().endswith('.png')]
  tf.logging.info('found %d images' % (len(image_list)))

  # iterate
  num_total_outputs = FLAGS.bsrn_recursions // FLAGS.bsrn_recursion_frequency
  for scale in scale_list:
    for image_name in image_list:
      tf.logging.info('- x%d: %s' % (scale, image_name))
      input_image_path = os.path.join(FLAGS.input_path, image_name)
      input_image = tf_image_read_session.run(tf_image_read, feed_dict={tf_image_read_path: input_image_path})
      output_images = model.upscale(input_list=[input_image], scale=scale)

      output_image_ensemble = np.zeros_like(output_images[0][0])
      ensemble_factor_total = 0.0
      
      for i in range(num_total_outputs):
        num_recursions = (i + 1) * FLAGS.bsrn_recursion_frequency
        output_image = output_images[i][0]

        ensemble_factor = 1.0 / (2.0 ** (num_total_outputs-num_recursions))
        output_image_ensemble = output_image_ensemble + (output_image * ensemble_factor)
        ensemble_factor_total += ensemble_factor
      
      output_image = output_image_ensemble / ensemble_factor_total

      output_image_path = os.path.join(FLAGS.output_path, 'x%d' % (scale), os.path.splitext(image_name)[0]+'.png')
      tf_image_save_session.run(tf_image_save_op, feed_dict={tf_image_save_path:output_image_path, tf_image_save_image:output_image})

  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  tf.app.run()