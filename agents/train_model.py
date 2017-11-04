from gym import wrappers
import numpy as np
from params import *
from models.model_predict import *
import time
np.random.seed(1234)

model=modelPredictor()
class LearnModel():
    def __init__(self):
        self.episode=1
        self.mode="test"
        if self.mode=="test":
            model.load()
        self.env=gym.make(env_name)
        self.clear()
        self.render=False
    def clear(self):
        self.observations=[]
        self.new_observations=[]
        self.actions=[]

    def train(self):
        model.train(self.observations,self.actions,self.new_observations,self.episode)
        self.clear()

    def test(self):
        model.test(self.observations,self.actions,self.new_observations)
        self.clear()

    def run_episode(self):
        observation = self.env.reset()
        t = 0   #to calculate time steps
        while True:
            if (self.render):
                self.env.render('rgb_array')#show the game, xlib error with multiple thread
            action=self.env.action_space.sample()
            self.observations.append(observation)
            self.actions.append(action)
            observation_new, reward, done, info = self.env.step(action)
            self.new_observations.append(observation_new)
            t=t+1
            observation=observation_new
            if self.mode=="test":
                self.test()
            if done:
                self.R_terminal=0
                #self.bellman_update() #can be used for batch
                print(" episode:",self.episode," took {} steps".format(t))
                if self.mode=="train":
                    self.train()
                break

    def run(self):
        start = time.time()
        if self.mode=="test":
            max_epochs=1
        while self.episode<=max_epochs:
            self.run_episode()
            self.episode+=1
            if self.episode%ckpt_episode==0:
                model.save(self.episode)
                print("saved model at episode {}".format(self.episode))
        end = time.time()
        print("took:",end - start)

if __name__=="__main__":
    modelLearner=LearnModel()
    modelLearner.run()
