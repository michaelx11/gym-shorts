import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os

import gym
env = gym.make('FrozenLake-v0')

# num observations
D = 16
# num actions
A = 4

# DQN setup heavily borrowed from: https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.edbzlkx5n
# What can I say, I need serious learning help

class Qnetwork():
    def __init__(self, h_size, identifier):
        print "init"
	# We'll go with fully connected layers
        self.scalarInput =  tf.placeholder(shape=[None,D],dtype=tf.float32)
        
        # V for Value network
        self.WV1 = tf.get_variable("WV1" + identifier, shape=[D, h_size],
                initializer=tf.contrib.layers.xavier_initializer())
        # rectified linear unit
        self.layerV1 = tf.nn.relu(tf.matmul(self.scalarInput,self.WV1))
        self.WV2 = tf.get_variable("WV2" + identifier, shape=[h_size, h_size],
                initializer=tf.contrib.layers.xavier_initializer())
        self.layerV2 = tf.nn.relu(tf.matmul(self.layerV1, self.WV2))
        self.WV3 = tf.get_variable("WV3" + identifier, shape=[h_size, A],
                initializer=tf.contrib.layers.xavier_initializer())
        # Value!
        self.Value = tf.matmul(self.layerV2,self.WV3)

        # A for Advantage network
        self.WA1 = tf.get_variable("WA1" + identifier, shape=[D, h_size],
                initializer=tf.contrib.layers.xavier_initializer())
        # rectified linear unit
        self.layerA1 = tf.nn.relu(tf.matmul(self.scalarInput,self.WA1))
        self.WA2 = tf.get_variable("WA2" + identifier, shape=[h_size, h_size],
                initializer=tf.contrib.layers.xavier_initializer())
        self.layerA2 = tf.nn.relu(tf.matmul(self.layerA1, self.WA2))
        self.WA3 = tf.get_variable("WA3" + identifier, shape=[h_size, A],
                initializer=tf.contrib.layers.xavier_initializer())
        # Advantage
        self.Advantage = tf.matmul(self.layerA2,self.WA3)
        
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.sub(self.Advantage,tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, A, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

def processState(states):
    l = np.zeros(D)
    l[states] = 1
    return l

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

batch_size = 30 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.01 #Final chance of random action
annealing_steps = 1000. #How many steps of training to reduce startE to endE.
num_episodes = 1000000 #How many episodes of game environment to train network with.
pre_train_steps = 100 #How many steps of random actions before training begins.
max_epLength = 100 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 128 #The size of the hidden layers
tau = 0.001 #Rate to update target network toward primary network

tf.reset_default_graph()
print 'init one'
mainQN = Qnetwork(h_size, 'main')
print 'init two'
targetQN = Qnetwork(h_size, 'target')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/annealing_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

rendering = False

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    sess.run(init)
    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
    for i in range(num_episodes):
        if total_steps >= 200000:
            rendering = True
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        #The Q-Network
        while j < max_epLength:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,2)
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            s1,r,d,_ = env.step(a)
            if rendering == True:
                env.render()
            if total_steps % 10000 == 0:
                print 'total steps', total_steps
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
            
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    
                    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
            rAll += r
            s = s1
            
            if d == True:

                break
        
        #Get all experiences from this episode and discount their rewards.
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        #Periodically save the model. 
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            print "Saved Model"
        if len(rList) % 10 == 0:
            print total_steps,np.mean(rList[-10:]), e
    saver.save(sess,path+'/model-'+str(i)+'.cptk')
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"

#rMat = np.resize(np.array(rList),[len(rList)/100,100])
#rMean = np.average(rMat,1)
#plt.plot(rMean)
