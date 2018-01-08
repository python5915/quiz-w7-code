"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')

    #根据论文最好将dropout的加上参数keep_prob=0.8，其中0.8是论文中作者实验取值
    #current = slim.dropout(current, keep_prob = 0.8,scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net

    #这是过渡层或者压缩层，其中pooling layer主要就是做bn,conv(1x1),avg_pool(2x2)
def transition_layer(net, num_outputs, kernel_size,scope='layer'):
    net = slim.batch_norm(net, scope = scope + '_bn')
    net = slim.conv2d(net, num_outputs, kernel_size, scope=scope + '_conv')
    net = slim.avg_pool2d(net, [2, 2], stride=2, padding='VALID', scope=scope + '_avg_pool')
    return net

def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:

            ##########################
            # my code:densenet的主要结构是输入图片->卷积(7x7conv,stride=2)+max池化(3x3conv,stride=2)
            #->第一个block->第一层pooling layer（过渡层）->继续第二个block和pooling layer ->继续重复->最后在卷积+softmax
            ##########################
            with slim.arg_scope(densenet_arg_scope()) as dsnet:
                logits = None
                #输入时，做第一次卷积
                #end_points字典存放每一步执行输出，以end_point为key，这是为了实现densenet(x` = H`([x0; x1 ... x`(l-1)]))
                end_point = 'input_conv'

                #参照论文的首次卷积，layers=16或者2*growth
                logits = slim.conv2d(images,2*growth,[7,7],stride=2,scope=end_point)
                end_points[end_point] = logits
                # 输入时，做第一次max_pool
                end_point = 'max_pool'
                logits = slim.max_pool2d(logits, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = logits

                #进入block层
                #block1
                end_point = 'block1'
                logits = block(logits,6,growth,end_point)
                end_points[end_point] = logits

                #进入transition layer，主要是bn+conv(1x1)+avg_pool(2X2)
                end_point = 'transition1'
                logits = transition_layer(logits,growth*compression_rate,[1,1],scope=end_point)
                end_points[end_point] = logits

                # block2
                end_point = 'block2'
                logits = block(logits, 12, growth, end_point)
                end_points[end_point] = logits
                # 进入transition layer2
                end_point = 'transition2'
                logits = transition_layer(logits, growth * compression_rate, [1, 1], scope=end_point)
                end_points[end_point] = logits
                # block3
                end_point = 'block3'
                logits = block(logits, 24, growth, end_point)
                end_points[end_point] = logits
                # 进入transition layer3
                end_point = 'transition3'
                logits = transition_layer(logits, growth * compression_rate, [1, 1], scope=end_point)
                end_points[end_point] = logits

                # block4
                end_point = 'block4'
                logits = block(logits, 16, growth, end_point)
                end_points[end_point] = logits

                end_point = 'last_dropout'
                logits = slim.dropout(logits, keep_prob=dropout_keep_prob, scope=end_point)
                end_points[end_point] = logits

                #做一次全局平均池化，kernel_size=上一层输出的weigt,height
                shape = logits.get_shape().as_list()
                kernel_size = [shape[1], shape[2]]
                logits = slim.avg_pool2d(logits, kernel_size, padding='VALID', scope='glogal_avg_pool')
                end_points['glogal_avg_pool'] = logits
                #1x1卷积
                logits = slim.conv2d(logits, 200, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                end_points['logits'] = logits

                logits = tf.squeeze(logits, [1, 2], name='squeeze')
                end_points['squeeze'] = logits

                #softmax分类
                end_points['predictions'] = slim.softmax(logits, scope='predictions')


    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
