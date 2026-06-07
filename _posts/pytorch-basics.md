# What is PyTorch?

PyTorch is a open source deep learning framework

# Why does PyTorch exist?

In order to understand this we need to take a quick look into common deep learning tasks.  

The example which we will take is a simple fully connected 3 layer network.  

### Step 1:
1. Input layer: Load the multidimensional data in batches. Let's denote the input vector as I.  
2. Hidden layer:  
- Multiply the weight matrix(W1) with the input layer. Let's say this generates an output vector A.   
- Apply an activation function (Relu) on the output vector A. This generates an ouput vector B.
3. Keep a track that each element of B = Relu(I*A)

### Step 2:
1. Multiply a second weight matrix(W2) with output of step 1 to generate the output vector. Let's say this generates an output vector Out. 
2. Apply a transformation (softmax) to convert the output vector into scalar value.  
3. Keep a track that each element of Out = (B*W2) 

### Step 3:
1. Compute the loss(Actual value - predicted value)
2. Use a optimization technique to minimize the loss(SGD, ADAM etc.). 
    - You can minimize the loss if you are able to adjust the weights(W1, W2) in such a way that the scalar value (the output of step 2) comes closer to the Actual value.
    - In order to do this you need to compute 

