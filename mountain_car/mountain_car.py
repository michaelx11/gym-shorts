import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import gym
env = gym.make('MountainCar-v0')

# observation = (pos, velocity) in ranges ([-1.2, 0.6], [-0.07, 0.07])
# action = 0, 1, or 2
# reward = -1, presumably 1 if we cross the flag

# Random actions :)
#env.reset()
#random_episodes = 0
#reward_sum = 0
#while random_episodes < 10:
#    env.render()
#    observation, reward, done, _ = env.step(np.random.randint(0,2))
#    print observation, reward, done
#    reward_sum += reward
#    if done:
#        random_episodes += 1
#        print "Reward for this episode was:",reward_sum
#        reward_sum = 0
#        env.reset()

# hyperparameters
H = 10 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-2 # feel free to play with this to train faster or more stably.
gamma = 0.96 # discount factor for reward

D = 2 # input dimensionality


tf.reset_default_graph()

#This defines the network as it goes from taking an observation of the environment to
#giving a probability of chosing to the action of moving left or right.
# MX: we create a placeholder of indefinite X D structure
observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
# Xavier initialization attempt to seed initial weights such that
# signals can propagate throughout the entire network but don't dominate.
# By choosing Weights from either uniform / gaussian with mean 0 and variance
# 1/ n_i (number of input neurons) we can math out that this preserves variance
# from input to output
W1 = tf.get_variable("W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
# rectified linear unit
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2", shape=[H, H],
           initializer=tf.contrib.layers.xavier_initializer())
layer2 = tf.nn.relu(tf.matmul(layer1, W2))
W3 = tf.get_variable("W3", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer2,W3)
probability = tf.nn.sigmoid(score)

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

# We want to pick a loss function that punishes lack of range of movement
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
W3Grad = tf.placeholder(tf.float32,name="batch_grad3")
batchGrad = [W1Grad, W2Grad, W3Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + 1 # all bad always
        discounted_r[t] = running_add
    return discounted_r

# if we're legitimately done, counts as zero loss
def zero_loss(r):
    return np.zeros_like(r)

#def reward_velocities(input_y):
#    SCALE = 10.0
#    """ take 1D float array of rewards and compute discounted reward """
#    velocities = input_y[:,1]
#    print velocities
#    discounted_r = np.zeros_like(velocities)
#    running_add = 0
#    for t in reversed(xrange(0, velocities.size)):
#        running_add = running_add + abs(SCALE * velocities[t])
#        discounted_r[t] = running_add
#    return discounted_r

xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 1000
MAX_STAGNANT_TICKS = 200
init = tf.initialize_all_variables()

ticks = 0
lastMaxPosChange = 0
finished_num = 0
# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    print 'hi'
    observation = env.reset() # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    maxPos = -10
    minPos = 10
    while episode_number <= total_episodes:
        ticks += 1

        # Rendering the environment slows things down,
        # so let's only look at it once our agent is doing a good job.
        if episode_number > 50 or rendering == True :
            env.render()
            rendering = True

        pos = observation[0]
        if (pos > maxPos):
          lastMaxPosChange = ticks
          maxPos = pos
        if (pos < minPos):
          lastMaxPosChange = ticks
          minPos = pos
        velocity = observation[1]
        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation,[1,D])
        # Run the policy network and get an action to take.
        tfprob = sess.run(probability,feed_dict={observations: x})
        # actions come from 0,3 (0 is left, 1 is neutral, 2 is right)
        action = 0 if np.random.uniform() < tfprob else 2

        xs.append(x) # observation
        y = 1 if action == 0 else 0 # a "fake label"
        ys.append(y)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        # We consider an episode done when we grind to a halt, we then do a backweighted
        if done or (ticks - lastMaxPosChange > MAX_STAGNANT_TICKS):
            if done:
                finished_num += 1
                print 'wooooo', finished_num
                if finished_num > 20:
                    rendering = True
            print 'episode: ' + str(episode_number)
            maxPos = -10
            minPos = 10
            lastMaxPosChange = 0 
            ticks = 0
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

            # compute the discounted reward backwards through time

            discounted_epr = discount_rewards(epr)
            # If done, we don't count it as loss
            if done:
                discounted_epr = zero_loss(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            if not done:
                discounted_epr /= np.std(discounted_epr)

            # Get the gradient for this episode, and save it in the gradBuffer
#            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad
            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1], W3Grad:gradBuffer[2]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print 'Average reward for episode %f.  Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size)

                if reward_sum/batch_size > 400:
                    print "Task solved in",episode_number,'episodes!'
                    break

                reward_sum = 0

            observation = env.reset()

print episode_number,'Episodes completed.'
