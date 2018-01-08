import tensorflow as tf

t = (1,2,3,4)

print(t[1])

shape=tf.placeholder(tf.float32, shape=[None, 3,3,3] )
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

shape = sess.run(
    [shape],
    feed_dict={shape:72})

print(shape)