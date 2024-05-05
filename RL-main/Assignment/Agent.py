import pickle
import numpy as np
import time


class Email_Marketing_Agent:
    def __init__(self, env):
        with open ('./q_tables/q_table_148-2104.bin', 'rb') as f:
            self.q_table = pickle.load(f)
        self.env = env
        
        
    def evaluate(self):
        """
        Evaluate the agent
        
        returns:
        ep_rewards[episode] (int): The reward
        ep_convs[episode]: The conversion
        respall (int): # of valid responses
        epT (float): runtime
        """
        # Hyperparameters
        maxepisodes = 1  # single pass through data

        # Performance metrics
        ep_rewards = np.zeros([maxepisodes])
        ep_convs = np.zeros([maxepisodes])

        episode = 0
        bestep = -9e9
        for episode in range(0, maxepisodes):  # eval across episodes
            done = False

            epstT = time.time()
            respall = 0
            sentall = 0
            while not done:  # eval within episode
                state = self.env.get_state()
                action = np.argmax(self.q_table[state, :]) # Exploit
                new_state, reward, nresp, nsent, done = self.env.step(action)

                ep_rewards[episode] += reward

                respall += nresp
                sentall += nsent

            ep_convs[episode] = respall / sentall

            epT = time.time() - epstT
            print('reward: {:.1f}, conv: {:.4f}, #resp: {}, time: {:.2f}'.\
                  format(ep_rewards[episode], ep_convs[episode], respall, epT))

            #Reset environment
            state = self.env.reset()

        print("\n===Evaluation completed.===\n")

        return ep_rewards[episode], ep_convs[episode], respall, epT

    
    def procsingle(self, state, utypidx, wdfrihol):    
        """
        Act on a single usertype and day type
        
        parameters:
        state (int): The state index
        utypidx (int): The usertype index.  A component of the state
        wdfrihol (int): The day type.  A component of the state
        
        returns:
        action (int): The action according to the policy
        conversion (float): The conversion rate
        """
        action = np.argmax(self.q_table[state, :])
        conversion = self.env.handle_single(action, utypidx, wdfrihol)
        
        return action, conversion
    