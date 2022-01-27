
import numpy as np
#import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import warnings
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))#1/(np.sqrt(2*np.pi)*sig)*

# RW
def RW_obs(s, isitgo, Q, h, beta): # find the probability that the agent will pick the "go" action
        
    #prob=0.5 # temporary value to avoid error when trying to print before assignment
    
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
    # uncomment the blwo after debugging
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            logpost += np.log(prior_prob_alpha)
        except Warning as e:
            print('We got a warning because we are trying to take the log of alpha=',prior_prob_alpha,'.')
            print('alpha_prior_est ',alpha_prior_est)
            
            quit()
    return logpost

def get_post(h,data,f_obs,alpha_prior_est):
    Q=np.zeros((4,2))
    prob_total=1
    for row in data:
        s=row[0]
        ai=act_i(row[1])
        r=row[2]
        prob_a = f_obs(s,ai,Q,h,beta)
        prob_total*=prob_a
        Q[s,ai] = RW_Q(Q[s,ai],r,h)
    prior_prob_alpha = gaussian(h,alpha_prior_est[0],alpha_prior_est[1]) 
    prob_total*=prior_prob_alpha
    return prob_total

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
def get2ndDerivBasic(f,x,step):
    rough1stderiv1 = (f(x) - f(x-step))/step
    rough1stderiv2 = (f(x+step) - f(x))/step
    rough2ndderiv = (rough1stderiv2-rough1stderiv1)/step
    return rough2ndderiv

def plot_this_in_3d(x,y,z,xlabel='x',ylabel='y',zlabel='z'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x,y)
    Z = z
    print(X.shape,Y.shape,Z.shape)
    surf = ax.plot_surface(X,Y,Z,cmap = cm.coolwarm, linewidth=0,antialiased=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
     

beta=10
### Hyperparameters ###

n = 1500 # number of subjects
n_iterations = 12
n_choices = 2000

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
#alpha_loc_est   = 0.7  # mean est for alpha
#alpha_scale_est = 1 # std dev est for alpha

starting_values_loc=np.arange(0.10,0.53,0.1)
starting_values_scale=[0.1]#np.arange(0.1,0.12,0.01)

def f(x):
    return 3*x**3

'''stepsizes=np.linspace(0.000001,10,10)
rough2nds = np.zeros(len(stepsizes))
for ii,size in enumerate(stepsizes):
    rough2nds[ii] = get2ndDerivBasic(f,10,size)
plt.figure()
plt.plot(stepsizes,rough2nds)
plt.show()
    
'''

all_data = gen_data(true_params,n_choices) # generate the data using the true parameters for each subject

alpharange = np.linspace(0.2,0.9999,20)

#### Do expectation maximisation ####
log_lik_max_prev=1000000000
for starting_val_scale in starting_values_scale:
    for starting_val_loc in starting_values_loc:
        alpha_prior_est = [starting_val_loc,starting_val_scale] # mean, std dev

        for k in range(n_iterations):
            m=[] 
            Sigma=[] 
            total_log_lik = 0
            ##plt.figure()
            print('We are on iteration ',k)
            for subj in range(n):
                data_for_subj = all_data[subj]
                bnds = [(0.1,0.9123)]
                
                init_ests = 0.5 # initial estimates for the parameters we're trying to find 
                ### minimize the negative log posterior ###
                argmax_est = minimize(neg_log_post,init_ests, args=(data_for_subj,RW_obs,alpha_prior_est),bounds=bnds)
                
                #print("The true alpha is ",alphas[subj],". The re-generated alpha={0}".format(argmax_est.x[0]),'\n')
                ##plt.show()

                m.append(argmax_est.x[0])
                total_log_lik += neg_log_post(m[-1],data_for_subj,RW_obs,alpha_prior_est)
                
                '''stepsizes=np.linspace(1e-6,1e-4,10)
                rough2nds = np.zeros(len(stepsizes))
                for ii,size in enumerate(stepsizes):
                    rough2nds[ii] = get2ndDeriv(neg_log_post,m[-1],data_for_subj,RW_obs,alpha_prior_est,size)
                plt.figure()
                plt.plot(stepsizes,rough2nds)
                plt.show()'''
                # test derivative code on known function
                #uncomment the line below after debug
                
                rough2ndderiv = get2ndDeriv(neg_log_post,m[-1],data_for_subj,RW_obs,alpha_prior_est,1e-4)
                #hess = nd.Hessian(neg_log_post)(m[-1],data_for_subj,RW_obs,alpha_prior_est)
                
                
                #det_hess = np.linalg.det(hess)
                #c = det_hess
                
                #Sigma.append(1/c)
                
                #uncomment the line below after debug
                Sigma.append(1/rough2ndderiv)
                #print('current sigma is ',1/rough2ndderiv)
                
            
            
            #bounds=Bounds([0.5,1.5],[1,2.5]) # set upper and lower bounds for the parameters we're trying to estimate. [lower_p1,lower_p2],[upper_p1,upper_p2]
            
            
            #############################
            #init_ests = [0.75,2.1] # initial estimates for the parameters we're trying to find 
            #argmax_est = minimize(neglogpost,init_ests, args=(rows,RW_obs,alpha_prior_est,rho_prior_est),bounds=bounds)
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

            print('the total log lik for iteration ',k,' is ',total_log_lik)
        #if total_log_lik<log_lik_max_prev:
        print('for starting value loc=',starting_val_loc,' and scale=',starting_val_scale)
        print('mean for alpha \t= ',mu)
        print('std dev for alpha \t=',nu)
        print('total log lik',total_log_lik)
        #log_lik_max_prev=total_log_lik


'''  
loc_range=np.arange(0.4,0.6,0.04)
scale_range=np.arange(0.01,0.2,0.04)
alpha_prior_range=[loc_range,scale_range]
z=np.zeros((len(loc_range),len(scale_range)))
for loci,loc in enumerate(alpha_prior_range[0]):
    for scalei,scale in enumerate(alpha_prior_range[1]):
        z[loci,scalei]=get_int_product(all_data,RW_obs,[loc,scale])
        print("int product for loc=",loc,"and scale=",scale,"is ",z[loci,scalei])

plot_this_in_3d(scale_range,loc_range,z,xlabel='stddev',ylabel='mean')
plt.show()
'''
plt.show()