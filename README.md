# Distributed_learning
Sample codes on distributed learning
1. Parameter averaging 
2. Asynchronous gradient descent method 

Profiling added : #from tensorflow.python.client import timeline. 

# Data Set 
IRIS data set . Split the Data set randomly into 2 and feeding it to the two worker nodes.

# Network Architecture : 
 2 Worker Node sessions have been created currently Each Node has below architecure 
 1. Input Layer 1 : 20 
 2. Hidden Layer : 10
 3. Output Layer : 10 
 
   Activation Relu :  
   Optimiser : Stochastic gradient Descent 
 
#  Parameter Averaging :
   Averaging done upon each weight update and over 100 epochs 
#  Asynchronous Stochastic Gradient Descent :
   Here I am using Gradient descent method updating the gradient by computing the meansquare loss over the entire data set 



# To be done 
: For both the sample algorithms distributed learning . We need worker processes in tensor flow running in parallel with global process. This has to be done in python . Current code does them in different sessions sequentially as I am not aware of how to create processes in python .

examples :https://stackoverflow.com/questions/42035400/parallel-processes-in-distributed-tensorflow

If I am not wrong the way I understand I need to instantiate 6 different local hosts that is for say 5 worker nodes and 1 global process . Now do a session.run on these hosts. 

So in Case of Asynchronous gradient descent : 
- > Worker: Each node completes gradient computation over all the data points i.e Loss = sum( square( label - output)) 

-> Worker : should suspend the process of the graident until it receives updates from Master 

-> Master : Should perfrom a single update and assign the weights back to the nodes and suspend until it receieves any further updates 

THESE are pending from basic version of coding point of view currently . 

   
