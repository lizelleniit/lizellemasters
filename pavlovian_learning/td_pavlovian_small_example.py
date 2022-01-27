### Lizelle Niit ###
### This is code to generate a simple example of classical conditioning. The agent over time 
### learns to associate a conditioned stimulus with a reward.


import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
import numpy as np

# note that indices in Python start at zero, so indices will be one lower than the trial 
# numbers they refer to 
alpha=0.5
gamma=1
Nsteps=5
Ntrials=20
light_on=1
num_weights=Nsteps-(light_on+1)
reward_arrives=4
x=np.zeros((num_weights,Nsteps))
w=np.zeros(num_weights)
V=np.zeros(Nsteps+1)
V_saved=np.zeros((Ntrials,Nsteps))
delta=np.zeros(Nsteps)
delta_saved=np.zeros((Ntrials,Nsteps))
r=np.zeros((Ntrials,Nsteps))

for triali in range(Ntrials): # Specify when the reward occurs. The current setup is very simple, a reward at the same time every trial.
	r[triali,reward_arrives]=1

# Fill the stimulus representation matrix with ones at time steps when the stimulus in question is present.
for i in range(num_weights):
	x[i,light_on+i+1]=1

for triali in range(Ntrials):
	for t in range(Nsteps):
		V[t]=np.sum(w*x[:,t]) # update value
		delta[t]=r[triali,t]+gamma*V[t+1]-V[t] # update prediction error
	for i in range(len(w)): # update weights
		w[i]=w[i]+alpha*np.sum(x[i,:]*delta) 
	delta_saved[triali]=delta # save data for plotting later
	#V_saved[triali]=V[:-1] # save data for plotting later
	print('V',V)
	print('delta',delta)
	print('w',w)

ax=plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(np.arange(1,Nsteps+1),delta_saved[0],'o-',label='trial 1')
plt.plot(np.arange(1,Nsteps+1),delta_saved[1],'o-',label='trial 2')
plt.plot(np.arange(1,Nsteps+1),delta_saved[6],'o-',label='trial 7')
plt.plot(np.arange(1,Nsteps+1),delta_saved[19],'o-',label='trial 20')
plt.xlabel('time step t')
plt.ylabel('prediction error δ')
plt.legend()

plt.show()
