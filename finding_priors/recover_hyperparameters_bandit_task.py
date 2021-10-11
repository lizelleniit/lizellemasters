import numpy as np
import numdifftools as nd
#import matplotlib.pyplot as plt
#import scipy.stats as stat
from scipy.optimize import minimize, Bounds
#from scipy.stats import norm
import warnings

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def neglogpost(h,*args):
    return -get_posterior(h,args[0],args[1])

def get_posterior(h,data,eps_prior_est):
    Q = np.zeros(4)
    logpost = 0
    N_act = np.zeros(4)
    for row in data: # go through all the actions for this subject
        
        act =row[1]
        r = row[2]
        N_act[act] += 1
        prob_act = get_prob_act(Q,h,act)
        
        logpost += np.log(prob_act)
        
        Q[act] += 1/(N_act[act])*(r - Q[act])
        #print('Q value for action ',act,' is ',Q[act])
    prior_prob_eps = gaussian(h[0],eps_prior_est[0],eps_prior_est[1]) 
        
    logpost += np.log(prior_prob_eps)
    return logpost
def get_prob_act(Q,h,act):
    if act == np.argmax(Q):
        prob_act = 1-h[0]+h[0]/4
    else:
        prob_act = h[0]/4
    return prob_act
def get_act(Q,h):
    eps = h[0]
    if np.random.random() > eps:
        act = np.argmax(Q)
        
    else:
        act = np.random.randint(4)
    return act
def get_reward(act):
    bandit_dists = np.array([[1,0.1],[-1,0.1],[0,0.1],[3,0.1]])
  
    r = np.random.normal(loc=bandit_dists[act,0],scale=bandit_dists[act,1])
    return r

def gen_data(true_params,n_choices):
    data = []
    for subj in range(n):
        N_act = np.zeros(4)
        #print("The generated parameters are alpha={0} and rho={1}".format(alphas[subj],rhos[subj]))
        ### generate dummy data ###
        
        data.append([])
        Q = np.zeros(4)
        # stimuli:
        eps=np.array([true_params[subj,0]]) # epsilon
        for i in range(n_choices):
            act = get_act(Q,eps)
            r = get_reward(act)
            N_act[act] += 1
            Q[act] += 1/N_act[act]*(r - Q[act])
            data[subj].append([9,act,r])
        
    return data

def get_Sigma(m,subj,rows,eps_prior_est):
    hess = nd.Hessian(neglogpost)(m[subj],rows,eps_prior_est)
        
    det_hess = np.linalg.det(hess)
    print('det hess',det_hess)
    c = det_hess
        
    Sigma = 1/c
    return Sigma
### Hyperparameters ###

n = 300 # number of subjects
n_iterations = 99
n_choices = 333

### I choose hyperparameters to use to generate the data ###

# true hyperparameters for alpha
eps_loc = 0.83 # mean for alpha
eps_scale = 0.02 # std dev for alpha



# generate an eps parameter for each subject
epss = []
for i in range(n):
    eps=-12345
    while eps<=0 or eps>1:
        print('eps',eps,'this is a possible problem')
        eps = np.random.normal(loc=eps_loc,scale=eps_scale)
    epss.append(eps)

# stack the alphas and the rhos together like (alpha1,rho1),(alpha2,rho2),...
true_params=np.stack([epss],axis=1)

#N_act = np.zeros(4)
#Q = np.zeros(4)
#dogs = []
#for i in range(333):
#    act = get_act(Q,[0.125])
#    r = get_reward(act)
#    N_act[act] += 1
#    Q[act] += 1/N_act[act]*(r - Q[act])
#    dogs.append([act,r])


#### Initial estimates for hyperparameters ####
eps_loc_est   = 0.5  # mean est for alpha
eps_scale_est = 0.3 # std dev est for alpha

eps_prior_est = [eps_loc_est,eps_scale_est] # mean, std dev

all_data = gen_data(true_params,n_choices) # generate the data using the true parameters for each subject

#### Do expectation maximisation ####
for k in range(n_iterations):
    m=[] 
    Sigma=[] 
    for subj in range(n):
        rows = all_data[subj]
        
        bounds=Bounds([0.000001],[0.9]) # set upper and lower bounds for the parameters we're trying to estimate. [lower_p1,lower_p2],[upper_p1,upper_p2]
    
        init_ests = [0.1] # initial estimates for the parameters we're trying to find 
        ### minimize the negative log posterior ###
        argmax_est = minimize(neglogpost,init_ests, args=(rows,eps_prior_est),bounds=bounds)
        
        m.append([argmax_est.x[0]])
        
        #print('true params for Subject ',subj,': ',true_params[subj])
        #print('estimated params for Subject ',subj,': ',m[subj])
        
        hess = nd.Hessian(neglogpost)(m[-1],rows,eps_prior_est)
        
        det_hess = np.linalg.det(hess)
        c = det_hess
        
                
        Sigma.append(1/c)
        #print('current sigma is ',1/c)
        
            
    ### now we do the global stuff, the stuff that combines the info for individual subjects    
    m=np.array(m)
    Sigma=np.array(Sigma)
    
    mu_arr = np.average(m,axis=0)
        
    nu_arr = np.sqrt(1/n*(np.sum(m**2+np.tile(Sigma,(1,1)).transpose(),axis=0))- mu_arr**2)
    
    
    eps_prior_est = [mu_arr[0],nu_arr[0]]

    print('mean for eps \t= ',mu_arr[0])
    print('std dev for eps \t=',nu_arr[0])
    
