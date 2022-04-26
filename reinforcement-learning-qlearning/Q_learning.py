import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

def greedy_Q_table(Q_table, obs, index = False):
    if index == True:
        return np.argmax((Q_table[obs, 0], Q_table[obs, 1], Q_table[obs, 2], Q_table[obs, 3]))
    else:
        return np.max((Q_table[obs, 0], Q_table[obs, 1], Q_table[obs, 2], Q_table[obs, 3]))

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)


    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):
            if random.uniform(0,1) > EPSILON:
                action = greedy_Q_table(Q_table, obs, index= True)
            else:
                action = np.random.randint(env.action_space.n)
            
            # update parameters
            next_obs, episode_reward, done, _ = env.step(action)
            
            # implement Q Learning
            if not done:
                Q_prime = greedy_Q_table(Q_table, next_obs)
                est = DISCOUNT_FACTOR * Q_prime
                actual = Q_table[obs, action]
                discount = episode_reward + est - actual
                Q_table[obs,action] += LEARNING_RATE * discount
            else:
                Q_table[obs, action] += LEARNING_RATE * (episode_reward - Q_table[obs, action])
            
            # update obs   
            obs = next_obs
            
            #action = env.action_space.sample() # currently only performs a random action.
            #obs,reward,done,info = env.step(action)
            #episode_reward += reward # update episode reward
        
        # decay Epsilon
        EPSILON *= EPSILON_DECAY
        
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 

        
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################