import tensorflow as tf
import numpy as np
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.python.client import timeline


# ======================
# Define the Graph
# ======================

# Define the Placeholders
# Define the weights for the layers

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=75, random_state=42)

def forwardprop(X,w1,w2,w3):
    Layer1 = tf.nn.relu(tf.matmul(X, w1))
    Layer2 = tf.nn.relu(tf.matmul(Layer1, w2))
    Layer3 = tf.nn.relu(tf.matmul(Layer2, w3))
    return Layer3

def param_avging(W11,W12,W13,W21,W22,W23):
    W1 = tf.divide(W11 + W21, 2)
    W2 = tf.divide(W12 + W22, 2)
    W3 = tf.divide(W13 + W23, 2)
    return W1,W2,W3

train_X, test_X, train_Y, test_Y = get_iris_data()
train_Y1 = train_Y
train_Y2 = test_Y
x1_size = train_X.shape[1]
x2_size = test_X.shape[1]# Number of input nodes: 4 features and 1 bias
y1_size = train_Y1.shape[1]  # Number of outcomes (3 iris flowers)
y2_size = train_Y2.shape[1]
print(train_Y1.shape,train_Y2.shape)


X1 = tf.placeholder("float", shape=[None, x1_size])
X2 = tf.placeholder("float", shape=[None, x2_size])
Y1 = tf.placeholder("float", shape=[None, y1_size])
Y2 = tf.placeholder("float", shape=[None, y2_size])


initial_Y1_layer_weights = np.random.rand(x1_size,20)
second_Y1_layer_weights = np.random.rand(20,10)
final_Y1_layer_weights = np.random.rand(10,y1_size)
initial_Y2_layer_weights = np.random.rand(x2_size,20)
second_Y2_layer_weights = np.random.rand(20,10)
final_Y2_layer_weights = np.random.rand(10,y2_size)


Y11_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y11", dtype="float32")
Y12_layer_weights = tf.Variable(second_Y1_layer_weights, name="share_Y12", dtype="float32")
Y13_layer_weights = tf.Variable(final_Y1_layer_weights, name="share_Y13", dtype="float32")
Y21_layer_weights = tf.Variable(initial_Y2_layer_weights, name="share_Y21", dtype="float32")
Y22_layer_weights = tf.Variable(second_Y2_layer_weights, name="share_Y22", dtype="float32")
Y23_layer_weights = tf.Variable(final_Y2_layer_weights, name="share_Y23", dtype="float32")

# Construct the Layers with RELU Activations

Y13_layer=forwardprop(X1, Y11_layer_weights, Y12_layer_weights, Y13_layer_weights)
Y23_layer = forwardprop(X2, Y21_layer_weights, Y22_layer_weights, Y23_layer_weights)


def computecostiter(Y,Y_layer):
    Y_Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits = Y_layer))
    return Y_Loss

Y1_Loss = computecostiter(Y1, Y13_layer)
Y2_Loss = computecostiter(Y2, Y23_layer)

predict_Y1 = tf.argmax(Y13_layer, axis=1)
predict_Y2 = tf.argmax(Y23_layer, axis=1)
# optimisers
optimi_Y1 = tf.train.GradientDescentOptimizer(0.01)
grads_vars_Y1 = optimi_Y1.compute_gradients(Y1_Loss, tf.trainable_variables())
#before doing a gradient update perform parameter avergaing or asynchronous stochastic gradient descent
updates_Y1 = optimi_Y1.apply_gradients(grads_vars_Y1)

optimi_Y2 = tf.train.GradientDescentOptimizer(0.01)
grads_vars_Y2 = optimi_Y2.compute_gradients(Y2_Loss, tf.trainable_variables())
#before doing a gradient update perform parameter avergaing or asynchronous stochastic gradient descent
updates_Y2 = optimi_Y2.apply_gradients(grads_vars_Y2)
# Joint Training
# Calculation (Session) Code
# ==========================

# open the session

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    #below is needed to create a timeline json file for profiling only!
     ####run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #####run_metadata = tf.RunMetadata()
    for epoch in range(2):
        # Train with each example
        for i in range(len(test_X)):
            start_time = time.time()
            session.run(updates_Y2, feed_dict={X2: test_X[i: i + 1], Y2: train_Y2[i: i + 1]})
            session.run(updates_Y1, feed_dict={X1: train_X[i: i + 1], Y1: train_Y1[i: i + 1]})
            duration = time.time() - start_time
            print("Duration taken by the indivisual Weight Updates = ",duration)
            start_time_2 = time.time()
            #Upon update call the paramter avergaing proceedure to
            [W1,W2,W3]=session.run(param_avging(Y11_layer_weights,Y12_layer_weights, Y13_layer_weights,
                                     Y21_layer_weights,Y22_layer_weights,Y23_layer_weights))
            session.run(tf.assign(Y11_layer_weights, W1))
            session.run(tf.assign(Y21_layer_weights, W1))
            session.run(tf.assign(Y12_layer_weights, W2))
            session.run(tf.assign(Y22_layer_weights, W2))
            session.run(tf.assign(Y13_layer_weights, W3))
            session.run(tf.assign(Y23_layer_weights, W3))
            duration_2 = time.time() - start_time_2
            print("Duration taken by the  Param_Aveging = ",duration_2)
            #tl = timeline.Timeline(run_metadata.step_stats)
            #ctf = tl.generate_chrome_trace_format()

        train_accuracy_Y1 = np.mean(np.argmax(train_Y1,axis=1) ==
                                session.run(predict_Y1, feed_dict={X1: train_X, Y1: train_Y1}))
        train_accuracy_Y2 = np.mean(np.argmax(train_Y2, axis=1) ==
                                 session.run(predict_Y2, feed_dict={X2: test_X, Y2: train_Y2}))
        print("Epoch = %d,train_accuracy of Y1 = %.2f%%,test accuracy of Second Node = %.2f%%"
                                  % (epoch + 1, 100. * train_accuracy_Y1, 100. * train_accuracy_Y2))
session.close()
