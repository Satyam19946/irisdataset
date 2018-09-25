#This program takes in input the Petal length, Petal width, Sepal length, Sepal width of a species of iris flowers and categorizes them as a Iris-Setosa, iris-Versicolor, or a Iris-Virginica. 

#Changing directory 
import os
os.chdir(r"C:\Users\Satyam\Desktop\Programs\Datasets\iris-species")

#importing required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style = 'darkgrid')
pd.options.display.max_rows = 10
my_data = pd.read_csv("Iris.csv") #Loading data in my_data. 

#Neural network with 1 hidden layer. First, second, and third layer consists of 4,10,3 neurons resp. Each layer is fully connected to the next layer. 

def modify_data(my_data): #Changing the y from 'Iris-Setosa'... to [0,1,2] where 0,1,2 corresponds to setosa,versicolor,virginica resp. 
    classes = np.array(['Iris-setosa','Iris-versicolor','Iris-virginica'])
    my_data = my_data.assign(Speciesnumbers = my_data['Species'].apply(lambda x: list(x == classes).index(True)))
    return my_data

my_data = modify_data(my_data) #Using the above function to modify data.
y_data = np.array(np.zeros([150,3]))

for i in range(len(my_data)): #Converting from 0,1,2 to [1,0,0],[0,1,0] and [0,0,1]
    y_data[i][my_data['Speciesnumbers'][i]] = 1

def interference(x,weight,bias):
    return tf.add(tf.matmul(x,weight),bias)

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy    

X = tf.placeholder(dtype = tf.float32, shape = [None,4])
y_true = tf.placeholder(dtype = tf.float32, shape = None)
weights1 = tf.Variable(tf.random_normal([4,10],stddev=0.5),name = 'weights1')  #connecting layer1 to layer2
bias1 = tf.Variable(tf.zeros(1),name = 'bias1') #bias in the first layer
weights2 = tf.Variable(tf.random_normal([10,3], stddev = 0.5),name = 'weights2') #connecting layer2 to layer3
bias2 = tf.Variable(tf.zeros(1),name = 'bias2') #bias in second layer.
layer2 = interference(X,weights1,bias1)
layer2 = tf.sigmoid(layer2)
layer3 = interference(layer2,weights2,bias2)
y_predict = tf.sigmoid(layer3)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits = y_predict))
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)
eval_op = evaluate(y_predict,y_true)
losses = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    X_data,y_data = my_data.iloc[:,1:5],y_data[:]
    for _ in range(10000):
        output = sess.run(loss,feed_dict = {X:X_data,y_true:y_data})
        sess.run(train,{X:X_data,y_true:y_data})
        print('loss =', output)
        losses.append(output)
        
plt.plot(losses)
plt.title("Loss vs Epochs")