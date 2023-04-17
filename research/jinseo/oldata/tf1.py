
import tensorflow as tf
import input_data
import threading
tf.compat.v1.disable_eager_execution()

# Training loop executed in each thread
def training_func():
  while True:
      batch = mnist.train.next_batch(100)
      global_step_val,_ = sess.run([global_step, train_step], feed_dict={x: batch[0], y_: batch[1]})
      print("global step: %d" % global_step_val)
      if global_step_val >= 20000:
        break

# create session and graph
sess = tf.compat.v1.Session()

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
global_step = tf.Variable(0, name="global_step")
y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

inc = global_step.assign_add(1)
with tf.control_dependencies([inc]):
  train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize graph and create mnist loader
sess.run(tf.compat.v1.global_variables_initializer())
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# create workers and execute threads
workers = []
for _ in range(4):
  t = threading.Thread(target=training_func)
  t.start()
  workers.append(t)

for t in workers:
  t.join()

# evaluate accuracy of the model
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels},
  session=sess))