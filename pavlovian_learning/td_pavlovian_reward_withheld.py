import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
alpha=0.3
gamma=1
Nsteps=120
Ntrials=100
light_on=40
#light_on_2=19
num_weights=Nsteps-(light_on+1)
#num_weights_2=Nsteps-light_on_2
reward_arrives=53
reward_withheld=29
x=np.zeros((num_weights,Nsteps))
#x2=np.zeros((num_weights_2,Nsteps))
w=np.zeros(num_weights)#w=np.zeros(num_weights+num_weights_2)
V=np.zeros(Nsteps+1)
V_saved=np.zeros((Ntrials,Nsteps))
delta=np.zeros(Nsteps)
delta_saved=np.zeros((Ntrials,Nsteps))
r=np.zeros((Ntrials,Nsteps))
for triali in range(Ntrials):
	#if (triali+1)%15!=0:
	#	r[triali,reward_arrives]=1
	if triali!=reward_withheld:
		r[triali,reward_arrives]=1
for i in range(num_weights):
	x[i,light_on+i+1]=1
#for i in range(num_weights_2):
#	x2[i,light_on_2+i]=1
#x_merged=np.concatenate((x,x2))
for triali in range(Ntrials):
	for t in range(Nsteps):
		V[t]=np.sum(w*x[:,t])
		delta[t]=r[triali,t]+gamma*V[t+1]-V[t]
		#w=w+alpha*x[:,t]*delta[t] # for online weight updates
	for i in range(len(w)): # for offline weight updates
		w[i]=w[i]+alpha*np.sum(x[i,:]*delta)
	delta_saved[triali]=delta
	V_saved[triali]=V[:-1]
	print('w',w)
	print('delta',delta)
	print('V',V)

ax=plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(np.arange(35,66),delta_saved[0,34:65],'o-',label='trial 1')
plt.plot(np.arange(35,66),delta_saved[29,34:65],'o-',label='trial 30')
plt.plot(np.arange(35,66),delta_saved[49,34:65],'o-',label='trial 50')
plt.plot(np.arange(35,66),delta_saved[99,34:65],'o-',label='trial 100')
plt.xlabel('time step t')
plt.ylabel('prediction error δ')
plt.legend()


fig = plt.figure()
ax = fig.gca(projection='3d')

steps = np.arange(35,66)
trials = np.arange(Ntrials)
steps, trials = np.meshgrid(steps, trials)

surf = ax.plot_surface(steps, trials, delta_saved[:,34:65], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('time step t')
ax.set_ylabel('trial number')
ax.set_zlabel('prediction error δ')

'''
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
steps = np.arange(Nsteps)
trials = np.arange(Ntrials)
steps, trials = np.meshgrid(steps, trials)

# Plot the surface.
surf = ax.plot_surface(steps, trials, V_saved, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('time step')
ax.set_ylabel('trial number')
ax.set_zlabel('Value V')


plt.show()
'''
plt.show()