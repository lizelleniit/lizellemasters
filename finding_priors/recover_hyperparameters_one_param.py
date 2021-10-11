import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import warnings


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))#1/(np.sqrt(2*np.pi)*sig)*

# RW
def RW_obs(s, isitgo, Q, h, beta): # find the probability that the agent will pick the "go" action
        
    prob=0.5 # temporary value to avoid error when trying to print before assignment
    
    #with warnings.catch_warnings():
    #    warnings.filterwarnings('error')
    #    try:
    #        thing =np.exp(Q[s,isitgo])/(np.exp(Q[s,0])+np.exp(Q[s,1]))
    #        #print('thing',thing,'; s: ',s,'; Q: ',Q)
    #    except Warning as e:
    #        print('WARNING: thing',thing,'; s: ',s,'; Q[s]: ',Q[s])
    #        quit()
    prob =np.exp(beta*Q[s,isitgo])/(np.exp(beta*Q[s,0])+np.exp(beta*Q[s,1]))
    return prob
    
def RW_Q(old_Q, r, h):    
    #print('oldQ',old_Q,'h',h,'r',r)
    return old_Q + h*(r - old_Q)

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

def neg_log_post(h,*args):
    return -get_log_post(h,args[0],args[1],args[2])
def get_log_post(h,data,f_obs,alpha_prior_est):
    Q = np.zeros((4,2))
    logpost = 0
    for row in data: # go through all the actions for this subject
        s = row[0]
        ai = act_i(row[1])
        r = row[2]
        
        prob_a = f_obs(s,ai,Q,h,beta)
        if prob_a<0:
            print("A probability smaller than 0!")
            break
        logpost += np.log(prob_a)
        
        Q[s,ai] = RW_Q(Q[s,ai],r,h)
    prior_prob_alpha = gaussian(h,alpha_prior_est[0],alpha_prior_est[1]) 
        
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            logpost += np.log(prior_prob_alpha)
        except Warning as e:
            print('We got a warning because we are trying to take the log of alpha=',prior_prob_alpha,'.')
            print('alpha_prior_est ',alpha_prior_est)
            
            quit()
    return logpost

def gen_data(true_params,n_choices):
    data = []
    for subj in range(n):
        #print("The generated parameters are alpha={0} and rho={1}".format(alphas[subj],rhos[subj]))
        data.append([])
        ### generate dummy data ###
        
        Q = np.zeros((4,2))
        # actions: go or nogo
        # stimuli:
        h=true_params[subj] # alpha
        for i in range(n_choices):
            
            stim = get_stim()          
            goprob=RW_obs(stim,1,Q,h,beta) # find the probability that the agent will "go"
            act = get_act(goprob) # given goprob, what action will the subject choose?
            r = get_reward(stim,act)
            
            Q[stim,act_i(act)] = Q[stim,act_i(act)] + h*(r - Q[stim,act_i(act)])#RW_Q(Q[stim,act_i(act)],r,h)
            data[subj].append([stim,act,r])

    return data

def get2ndDeriv(f,x,data_for_subj,f_obs,prior,step):
    rough1stderiv1 = (f(x,data_for_subj,f_obs,prior) - f(x-step,data_for_subj,f_obs,prior))/step
    rough1stderiv2 = (f(x+step,data_for_subj,f_obs,prior) - f(x,data_for_subj,f_obs,prior))/step
    rough2ndderiv = (rough1stderiv2-rough1stderiv1)/step
    return rough2ndderiv
        

beta=100
### Hyperparameters ###

n = 40 # number of subjects
n_iterations = 10000
n_choices = 333

### I choose hyperparameters to use to generate the data ###

# true hyperparameters for alpha
alpha_loc = 0.5123456 # mean for alpha
alpha_scale = 0.1123456 # std dev for alpha

# generate an alpha parameter for each subject
alphas = []
for i in range(n):
    alpha=-1
    while alpha<=0 or alpha>1:
        if alpha!=-1:
            print('this value of alpha of ',alpha,' is a possible problem')
        alpha = np.random.normal(loc=alpha_loc,scale=alpha_scale)
    alphas.append(alpha)

# stack the alphas and the rhos together like (alpha1,rho1),(alpha2,rho2),...
true_params=alphas # gets less trivial when there's more than one param

#### Initial estimates for hyperparameters ####
alpha_loc_est   = 0.7  # mean est for alpha
alpha_scale_est = 1 # std dev est for alpha

alpha_prior_est = [alpha_loc_est,alpha_scale_est] # mean, std dev


all_data = gen_data(true_params,n_choices) # generate the data using the true parameters for each subject

alpharange = np.linspace(0.5,0.9999,20)
#### Do expectation maximisation ####
for k in range(n_iterations):
    m=[] 
    Sigma=[] 
    #plt.figure()
    total_log_lik = 0
    for subj in range(n):
        data_for_subj = all_data[subj]
        bnds = [(0.1,0.9123)]
        
        init_ests = 0.7 # initial estimates for the parameters we're trying to find 
        ### minimize the negative log posterior ###
        argmax_est = minimize(neg_log_post,init_ests, args=(data_for_subj,RW_obs,alpha_prior_est),bounds=bnds)
        
        ### plot neg_log_post for a range of parameter values 
        #postvals = np.zeros(len(alpharange))
        #for ipostval,alph in enumerate(alpharange):
        #    postvals[ipostval] = neg_log_post(alph,data_for_subj,RW_obs,alpha_prior_est)
        #plt.plot(alpharange,postvals)

        #print("The re-generated parameters are alpha={0} and rho={1}".format(argmax_est.x[0],argmax_est.x[1]),'\n')
        m.append(argmax_est.x[0])
        total_log_lik += neg_log_post(m[-1],data_for_subj,RW_obs,alpha_prior_est)
        #stepsizes=np.linspace(0.0001,0.5,10)
        #rough2nds = np.zeros(len(stepsizes))
        #for ii,size in enumerate(stepsizes):
        #    rough2ndderiv = get2ndDeriv(neg_log_post,m[-1],data_for_subj,RW_obs,alpha_prior_est,size)
        #    rough2nds[ii] = rough2ndderiv
        #plt.plot(stepsizes,rough2nds)
        # test derivative code on known function
        rough2ndderiv = get2ndDeriv(neg_log_post,m[-1],data_for_subj,RW_obs,alpha_prior_est,0.01)
        #hess = nd.Hessian(neg_log_post)(m[-1],data_for_subj,RW_obs,alpha_prior_est)
        
        '''
        det_hess = np.linalg.det(hess)
        c = det_hess
        
        Sigma.append(1/c)
        
        '''
        Sigma.append(1/rough2ndderiv)
        #print('current sigma is ',1/rough2ndderiv)
        
        
    
    ### now we do the global stuff, the stuff that combines the info for individual subjects    
    m=np.array(m)
    Sigma=np.array(Sigma)
    mu = np.average(m,axis=0)
    
    # The purpose of the tile and transpose functions below is to make sigma the same shape as m. For a given subject, sigma is the same for all the parameters. (todo: double check that) 
    nu = np.sqrt(1/n*np.sum((m**2+Sigma),axis=0)- mu**2)
    
    # mu_arr: [mean of the prior gaussian for alpha,mean of the prior gaussian for rho]
    # nu_arr: [std dev of the prior gaussian for alpha,std dev of the prior gaussian for rho]
    # these means and std devs are now our new estimates for the hyperparameters
    alpha_prior_est = [mu,nu]

    print('mean for alpha \t= ',mu)
    print('std dev for alpha \t=',nu)
    print('total log lik',total_log_lik)

    plt.show()