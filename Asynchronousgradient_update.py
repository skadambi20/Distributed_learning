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

def gradient_avging(gradW1,gradW2):
    Yg1_layer_weights = Yg1_layer_weights - 0.8*(tf.divide(gradW1[0]+gradW2[0],2))
    Yg2_layer_weights = Yg2_layer_weights - 0.8*(tf.divide(gradW1[1]+gradW2[1],2))
    Yg3_layer_weights = Yg3_layer_weights - 0.8*(tf.divide(gradW1[2]+gradW2[2],2))
    return

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

global Yg1_layer_weights,Yg2_layer_weights,Yg3_layer_weights
#global parameters
Yg1_layer_weights = tf.Variable(initial_Y1_layer_weights, name="global_Y1", dtype="float32")
Yg2_layer_weights = tf.Variable(second_Y1_layer_weights, name="global_Y2", dtype="float32")
Yg3_layer_weights = tf.Variable(final_Y1_layer_weights, name="global_Y3", dtype="float32")

# Construct the Layers with RELU Activations

#Node 1
Yg11_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y11", dtype="float32")
Yg12_layer_weights = tf.Variable(second_Y1_layer_weights, name="share_Y12", dtype="float32")
Yg13_layer_weights = tf.Variable(final_Y1_layer_weights, name="share_Y13", dtype="float32")
# Node 2
Yg21_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y21", dtype="float32")
Yg22_layer_weights = tf.Variable(second_Y1_layer_weights, name="share_Y22", dtype="float32")
Yg23_layer_weights = tf.Variable(final_Y1_layer_weights, name="share_Y23", dtype="float32")

Y13_layer=forwardprop(X1, Yg11_layer_weights, Yg12_layer_weights, Yg13_layer_weights)
Y23_layer = forwardprop(X2, Yg21_layer_weights, Yg22_layer_weights, Yg23_layer_weights)


def computecostiter(Node1):
    if(Node1 == True):
        Y_Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y1, logits = Y13_layer))
    else:
        Y_Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y2, logits=Y23_layer))
    return Y_Loss


Y1_Loss = computecostiter(True)
Y2_Loss = computecostiter(False)

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
    init = tf.global_variables_initializer()
    init_1  = tf.local_variables_initializer()
    session.run(init)
    session.run(init_1)
    #below is needed to create a timeline json file for profiling only!
     ####run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #####run_metadata = tf.RunMetadata()
    #session.run(updates_Y1,feed_dict = {X1:test_X[0:1],Y1:train_Y2[0:1]})
    #for epoch in range(2):
        # Train with each example
    for epoch in range(100):
    #do one forward propagation

    #compute the gradient on all the data points which is 75 ?
        grad_vals_1 = session.run([grad[1] for grad in grads_vars_Y1 ])
        grad_vals_2 = session.run([grad[1] for grad in grads_vars_Y2])
    #upon computing the gradients on both the nodes now compute the average of the graident and update the weights in each
        [W1,W2,W3] = session.run(gradient_avging(grad_vals_1,grad_vals_2))
        tf.assign(Y11_layer_weights,W1)
        tf.assign(Y12_layer_weights, W2)
        tf.assign(Y13_layer_weights, W3)
        train_accuracy_Y1 = np.mean(np.argmax(train_Y1, axis=1) ==
                                session.run(predict_Y1, feed_dict={X1: train_X, Y1: train_Y1}))
        train_accuracy_Y2 = np.mean(np.argmax(train_Y2, axis=1) ==
                                session.run(predict_Y2, feed_dict={X2: test_X, Y2: train_Y2}))
        print("Epoch = %d,train_accuracy of Y1 = %.2f%%,test accuracy of Second Node = %.2f%%"
          % (epoch + 1, 100. * train_accuracy_Y1, 100. * train_accuracy_Y2))




          # session.run(Y1_Loss, feed_dict={X1: train_X[i: i + 1], Y1: train_Y1[i: i + 1]})

session.close()
