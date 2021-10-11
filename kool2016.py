### Some of the code for the agent classes was taken from
### the Berkeley CS188 reinforcement learning tut at ...


import numpy as np 
import csv
import matplotlib.pyplot as plt



class Environment():

    def __init__(self):
        self.rewards = {}
        self.rewards['stateA']={}
        self.rewards['stateB']={}
        self.rewards['stateC']={}
        self.rewards['stateA']['actionA']=0
        self.rewards['stateA']['actionB']=0
        self.rewards['stateB']['actionA']=0.74
        self.rewards['stateB']['actionB']=0.7
        self.rewards['stateC']['actionA']=0.27
        self.rewards['stateC']['actionB']=0.26
        self.states=['stateA','stateB','stateC']

    def getReward(self,state,action):
        return self.rewards[state][action]

    def updateRewards(self):
        self.rewards['stateB']['actionA'] = self.updateReward(self.rewards['stateB']['actionA'])
        self.rewards['stateB']['actionB'] = self.updateReward(self.rewards['stateB']['actionB'])
        self.rewards['stateC']['actionA'] = self.updateReward(self.rewards['stateC']['actionA'])
        self.rewards['stateC']['actionB'] = self.updateReward(self.rewards['stateC']['actionB'])
        return

    def updateReward(self,reward):
        rand = np.random.normal(loc=0,scale=0.025)
        if reward+rand > 0.75 or reward+rand < 0.25:
            reward = reward-rand
        else:
            reward+=rand
        if reward<0.25 or reward>0.75:
            print("the reward has reach an unpermissible value. the programmer goofed.")
        return reward

    def getNextState(self,state,action):
        if state=='stateA':
            rand = np.random.random()
            if action=='actionA':
                if rand<1:
                    nextState='stateB'
                else:
                    nextState='stateC'
            elif action=='actionB':
                if rand<1:
                    nextState='stateC'
                else:
                    nextState='stateB'
        elif state=='stateB' or state=='stateC':
            nextState='terminal'
        return nextState
    def getNextStateAndReward(self,state,action):
        sp = self.getNextState(state,action)
        r = self.getReward(state,action)
        
        #self.updateRewards()        
        return sp,r

class Agent():
    def __init__(self):
        self.Q={}
        self.actions = ['actionA','actionB']

    def getQValue(self,state,action):
        ai = self.actions.index(action)
        if state in self.Q:
            return self.Q[state][ai]
        else:
            return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        
        Qfors=np.zeros(len(self.actions)) # make an empty array of Q values for this state
        for ai,a in enumerate(self.actions): # for each action, fill in its Q value
            Qfors[ai]=self.getQValue(state,a) 
        best_Q=np.max(Qfors) # pick the highest Q value
        
        return best_Q

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        Qfors=np.zeros(len(self.actions))
        for ai,a in enumerate(self.actions):
                
            Qfors[ai]=self.getQValue(state,a)
        best_ai=np.random.choice(np.where(Qfors==Qfors.max())[0])
       
        return self.actions[best_ai]

    def getAction(self, state):
        best_a=self.computeActionFromQValues(state)
        
        if np.random.random()<self.epsilon:
            # return a random action
            return np.random.choice(self.actions)
        else: # return the best action
            return best_a 

        return action
        
class QLearningAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.epsilon = 0.2
        self.discount = 0.9
        self.alpha = 0.5
        

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        ai=self.actions.index(action)
        # check whether we have a key for "state"
        if not state in self.Q:
            self.Q[state]=np.zeros(len(self.actions))
        if not nextState in self.Q:
            #print('nextState ',nextState)
            self.Q[nextState]=np.zeros(len(self.actions))
        
        if nextState=='terminal':
            maxQ=0
        else:
            maxQ=self.computeValueFromQValues(nextState)
        #print('Q[',state,'][',ai,']: ',self.Q[state][ai])
        self.Q[state][ai]+=self.alpha*(reward + self.discount*maxQ-self.getQValue(state,action))
        
        
        return 

class SARSAAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.epsilon = 0.2
        self.discount = 0.9
        self.alpha = 0.5
        

    def update(self, state, action, nextState, nextAction, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        ai=self.actions.index(action)
        # check whether we have a key for "state"
        if not state in self.Q:
            self.Q[state]=np.zeros(len(self.actions))
        if not nextState in self.Q:
            #print('nextState ',nextState)
            self.Q[nextState]=np.zeros(len(self.actions))
        
        #print('Q[',state,'][',ai,']: ',self.Q[state][ai])
        self.Q[state][ai]+=self.alpha*(reward + self.discount*self.getQValue(nextState,nextAction)-self.getQValue(state,action))
        
        
        return 

class SARSALambdaAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.epsilon = 0.2
        self.discount = 0.9
        self.alpha = 0.5
        self.lamb=0
        

    def update(self, state, action, nextState, nextAction, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        ai=self.actions.index(action)
        # check whether we have a key for "state"
        if not state in self.Q:
            self.Q[state]=np.zeros(len(self.actions))
        if not nextState in self.Q:
            #print('nextState ',nextState)
            self.Q[nextState]=np.zeros(len(self.actions))
        
        #print('Q[',state,'][',ai,']: ',self.Q[state][ai])
        self.Q[state][ai]+=self.alpha*(reward + self.discount*self.getQValue(nextState,nextAction)-self.getQValue(state,action))
        
        
        return 


env = Environment()
agent = QLearningAgent()
sarsaAgent = SARSAAgent()
sarsaLambdaAgent = SARSALambdaAgent()
'''# Q learning 
for epi in range(500): 
    #print('Episode ',epi)
    s = 'stateA'
    while s!='terminal':
        a = agent.getAction(s)
        sp,r = env.getNextStateAndReward(s,a)
        #print('we are in state s=',s,' and take action a=',a)
        #print('we move to state sp=',sp,' and get reward r=',r)
        agent.update(s,a,sp,r)
        s = sp


print('Q',agent.Q)
'''

# SARSA
for epi in range(1000): 
    #print('Episode ',epi)
    s = 'stateA'
    a = sarsaAgent.getAction(s)
    while s!='terminal':
        
        sp,r = env.getNextStateAndReward(s,a)
        ap = sarsaAgent.getAction(sp)
        #print('we are in state s=',s,' and take action a=',a)
        #print('we move to state sp=',sp,' and get reward r=',r)
        sarsaAgent.update(s,a,sp,ap,r)
        s = sp
        a = ap

print(sarsaAgent.Q)


# SARSA lambda learning

n_epi_sarsa_lambda=1000
delta = np.zeros((2,n_epi_sarsa_lambda))
eDict={}
for state in env.states:
    eDict[state]={}
    eDict[state]['actionA']=np.zeros((2,n_epi_sarsa_lambda))
    eDict[state]['actionB']=np.zeros((2,n_epi_sarsa_lambda))

    sarsaLambdaAgent.Q[state]=np.zeros(2)
for epi in range(n_epi_sarsa_lambda):
    s = 'stateA'
    a = sarsaLambdaAgent.getAction(s)
    
    for i in range(1,3): # iterate through our two stages, 1 and 2
        sp,r = env.getNextStateAndReward(s,a)
        ap = sarsaLambdaAgent.getAction(sp)
        ai=sarsaLambdaAgent.actions.index(a)
        if i==1:
            elig=0
        elif i==2:
            elig=eDict[s][a][i-1,epi]
        eDict[s][a][i-1,epi]=elig+1 # these indices are confusing because python starts at 0 and Kool 2016 at 1. make sure it's right.
        delta[i-1,epi]=r+sarsaLambdaAgent.getQValue(sp,ap)-sarsaLambdaAgent.getQValue(s,a)
        sarsaLambdaAgent.Q[s][ai]=sarsaLambdaAgent.getQValue(s,a)+sarsaLambdaAgent.alpha*delta[i-1,epi]*eDict[s][a][i-1,epi]
        # now we decay all eligibilities by lambda:
        for state in eDict:
            for action in eDict[state]:
                eDict[state][action]*=sarsaLambdaAgent.lamb
        s=sp
        a=ap

print(sarsaLambdaAgent.Q)
