import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import argparse
import csv
import scipy.stats as stat
from scipy.optimize import minimize, Bounds
from scipy.integrate import quad
from scipy.stats import norm

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# RW
def RW_obs(s, isitgo, Q, h): # find the probability that the agent will pick the "go" action
    #print('s',s)
    thing =np.exp(Q[s,isitgo])/(np.exp(Q[s,0])+np.exp(Q[s,1]))
    #if thing==0:
    #    print("Q",Q[s,isitgo])
    #    print("thing",thing)
    return thing



def RW_Q(old_Q, r, h):
    return old_Q + h[0]*(h[1]*r - old_Q)

def get_reward(stim,act):
    
    if stim == 0:
        if act == "go":
            r = get_r_prob_80()
        elif act == "nogo":
            r = 0
        
    elif stim == 1:
        if act == "go":
            r = 0
        elif act == "nogo":
            r = get_r_prob_80()
        
    elif stim == 2:
        if act == "go":
            r = get_loss_prob_80()
        elif act == "nogo":
            r = 0
        
    elif stim == 3:
        if act == "go":
            r = 0
        elif act == "nogo":
            r = get_loss_prob_80()
    return r
def get_r_prob_80():
    rand = np.random.random()
    if rand < 0.8:
        return 1
    else:
        return 0
def get_loss_prob_80():
    rand = np.random.random()
    if rand < 0.8:
        return -1
    else:
        return 0
def get_stim(): # generates a state
    stim = np.random.randint(4)
    
    return stim
def get_act(probgo):
    ran = np.random.random()
    if ran<probgo:
        # choose go
        act = "go"
    else:
        act = "nogo"
    return act
def act_i(act):
    if act == "go":
        return 1
    else:
        return 0
def negloglik(h,*args):
    return -get_loglik(h,args[0],args[1])
def get_loglik(h,data,f_obs):
    Q = np.zeros((4,2))
    loglik = 0
    for row in data:
        s = row[0]
        ai = act_i(row[1])
        r = row[2]
        
        prob_a = f_obs(s,ai,Q,h)
        if prob_a<0:
            print("A probability smaller than 0!")
            break
        loglik += np.log(prob_a)
        Q[s,ai] = RW_Q(Q[s,ai],r,h)
    return loglik

### Hyperparameters ###

n = 5 # number of subjects

# hyperparameters for alpha
alpha_loc = 0.5
alpha_scale = 0.1
# hyperparameters for rho
rho_loc = 1
rho_scale = 0.1
#######################

### Generate the parameters from the hyperparameters ###

# alpha
alphas = []
for i in range(n):
    alpha=-1
    while alpha<=0 or alpha>1:
        alpha = np.random.normal(loc=alpha_loc,scale=alpha_scale)
    alphas.append(alpha)
# rho
rhos = []
for i in range(n):
    rho=-1
    while rho<=0:
        rho = np.random.normal(loc=rho_loc,scale=rho_scale)
    rhos.append(rho)

true_params=np.stack([alphas,rhos],axis=1)


est_params=[]
for subj in range(n):
    print("We are now on subject ",subj)
    #print("The generated parameters are alpha={0} and rho={1}".format(alphas[subj],rhos[subj]))
    ### generate dummy data ###

    f = open('dummydata'+str(subj)+'.txt','w')
    f.write("s\t")
    f.write("a\t")
    f.write("r\n")
    Q = np.zeros((4,2))
    # actions: go or nogo
    # stimuli:
    h=np.array([alphas[subj],rhos[subj]]) # alpha, rho
    for i in range(10000):
        stim = get_stim()
        #print('The current stimulus is ',stim)
        goprob=RW_obs(stim,1,Q,h) # find the probability that the agent will "go"
        #print('The prob of choosing go is ',goprob)
        act = get_act(goprob) # plug the "go" probability into the get_act function
    
        r = get_reward(stim,act)
        #print('r',r)
        q = Q[stim,act_i(act)]
        #print('current Q is ',q)
        #print('current Q gets changed by ',h[0]*(h[1]*r - Q[stim,act_i(act)]))
        Q[stim,act_i(act)] = Q[stim,act_i(act)] + h[0]*(h[1]*r - Q[stim,act_i(act)])#RW_Q(Q[stim,act_i(act)],r,h)
        #print(Q)
        f.write(str(stim))
        f.write('\t')
        f.write(act)
        f.write('\t')
        f.write(str(r))
        f.write('\n')
    f.close()
    ##################################


    ### Read the data from file ###

    with open('dummydata'+str(subj)+'.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        rows = [[int(row[0]),row[1], int(row[2])] for row in reader]
    ###############################

    ### get the log likelihood ###

    Q = np.zeros((4,2))
    bounds=Bounds([0,0],[1,3])
    loglik = get_loglik(h,rows,RW_obs)
    #############################

    ### minimize the log likelihood ###
    something = minimize(negloglik,[0.5,1], args=(rows,RW_obs),bounds=bounds)
    #print("The re-generated parameters are alpha={0} and rho={1}".format(something.x[0],something.x[1]),'\n')
    ###################################
    est_params.append([something.x[0],something.x[1]])
est_params=np.array(est_params)
print(true_params)
print(est_params)
### Try to recover the hyperparams ###
mean_alpha,std_alpha=norm.fit(est_params[:,0])
mean_rho,std_rho=norm.fit(est_params[:,1])

print("The mean of alpha is ",mean_alpha)
print("The std dev of alpha is ",std_alpha)
print("The mean of rho is ",mean_rho)
print("The std dev of rho is ",std_rho)

n_bins=10
plt.figure()
plt.hist(est_params[:,0], bins=n_bins)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
y = norm.pdf(x, mean_alpha, std_alpha)
plt.plot(x, y)
plt.xlabel('alpha')


plt.figure()
plt.hist(est_params[:,1], bins=n_bins)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
y = norm.pdf(x, mean_rho, std_rho)
plt.plot(x, y)
plt.xlabel('rho')



plt.show()
