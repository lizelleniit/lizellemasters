
import numpy as np
import numdifftools as nd

import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import warnings
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))#1/(np.sqrt(2*np.pi)*sig)*

# RW
def RW_obs(s, isitgo, Q, h): # find the probability that the agent will pick the "go" action
        
    #prob=0.5 # temporary value to avoid error when trying to print before assignment
    
    #with warnings.catch_warnings():
    #    warnings.filterwarnings('error')
    #    try:
    #        thing =np.exp(Q[s,isitgo])/(np.exp(Q[s,0])+np.exp(Q[s,1]))
    #        #print('thing',thing,'; s: ',s,'; Q: ',Q)
    #    except Warning as e:
    #        print('WARNING: thing',thing,'; s: ',s,'; Q[s]: ',Q[s])
    #        quit()
    beta=h[1]
    prob =np.exp(beta*Q[s,isitgo])/(np.exp(beta*Q[s,0])+np.exp(beta*Q[s,1]))
    
    return prob
    
def RW_Q(old_Q, r, h):    
    alpha=h[0]
    return old_Q + alpha*(r - old_Q)

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

def log_post(h,*args):
    return get_log_post(h,args[0],args[1],args[2],args[3])
def neg_log_post(h,*args):
    
    return -get_log_post(h,args[0],args[1],args[2],args[3])
def get_log_post(h,data,f_obs,alpha_prior_est,beta_prior_est):
    Q = np.zeros((4,2))
    logpost = 0
    for row in data: # go through all the actions for this subject
        s = row[0]
        ai = act_i(row[1])
        r = row[2]
        
        prob_a = f_obs(s,ai,Q,h)
        if prob_a<0:
            print("A probability smaller than 0!")
            break
        logpost += np.log(prob_a)
        
        Q[s,ai] = RW_Q(Q[s,ai],r,h)
    prior_prob_alpha = gaussian(h[0],alpha_prior_est[0],alpha_prior_est[1]) 
    prior_prob_beta = gaussian(h[1],beta_prior_est[0],beta_prior_est[1]) 
    #print('alpha_prior_est ',alpha_prior_est)
    #print('beta_prior_est ',beta_prior_est)
    #print('h',h,'; prior_prob_alpha ',prior_prob_alpha,'; prior_prob_beta ',prior_prob_beta)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            logpost += np.log(prior_prob_alpha)
        except Warning as e:
            print('We got a warning because we are trying to take the log of prior_prob_alpha=',prior_prob_alpha,'.')
            print('alpha_prior_est ',alpha_prior_est)
            print('h',h)
            
            quit()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            logpost += np.log(prior_prob_beta)
        except Warning as e:
            print('We got a warning because we are trying to take the log of prior_prob_beta=',prior_prob_beta,'.')
            print('beta_prior_est ',beta_prior_est)
            print('h',h)
            quit()
    return logpost

'''def get_post(h,data,f_obs,alpha_prior_est):
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
'''
def gen_data(true_params,n_choices,n):
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
            goprob=RW_obs(stim,1,Q,h) # find the probability that the agent will "go"
            act = get_act(goprob) # given goprob, what action will the subject choose?
            r = get_reward(stim,act)
            
            Q[stim,act_i(act)] = Q[stim,act_i(act)] + h[0]*(r - Q[stim,act_i(act)])#RW_Q(Q[stim,act_i(act)],r,h)
            data[subj].append([stim,act,r])

    return data
def get1stDerivMultiVar(f,x_vec,changing_i,data_for_subj,f_obs,prior,beta_prior,step):
    upper_x=x_vec.copy()
    upper_x[changing_i]+=step/2
    lower_x=x_vec.copy()
    lower_x[changing_i]-=step/2
    return (f(upper_x,data_for_subj,f_obs,prior,beta_prior)-f(lower_x,data_for_subj,f_obs,prior,beta_prior))/step
def get1stDerivMultiVarBelow(f,x_vec,changing_i,data_for_subj,f_obs,prior,beta_prior,step):
    upper_x=x_vec.copy()
    lower_x=x_vec.copy()
    lower_x[changing_i]-=step
    return (f(upper_x,data_for_subj,f_obs,prior,beta_prior)-f(lower_x,data_for_subj,f_obs,prior,beta_prior))/step
def get1stDerivMultiVarAbove(f,x_vec,changing_i,data_for_subj,f_obs,prior,beta_prior,step):
    upper_x=x_vec.copy()
    upper_x[changing_i]+=step
    lower_x=x_vec.copy()
    return (f(upper_x,data_for_subj,f_obs,prior,beta_prior)-f(lower_x,data_for_subj,f_obs,prior,beta_prior))/step

def get2ndDerivMultiVar(f,x_vec,changing_i,data_for_subj,f_obs,prior,beta_prior,step):
    firstDerivAbove=get1stDerivMultiVarAbove(f,x_vec,changing_i,data_for_subj,f_obs,prior,beta_prior,step)
    firstDerivBelow=get1stDerivMultiVarBelow(f,x_vec,changing_i,data_for_subj,f_obs,prior,beta_prior,step)
    return (firstDerivAbove-firstDerivBelow)/step

def get2ndDeriv(f,x,data_for_subj,f_obs,prior,beta_prior,step):
    rough1stderiv1 = (f(x,data_for_subj,f_obs,prior,beta_prior) - f(x-step,data_for_subj,f_obs,prior,beta_prior))/step
    rough1stderiv2 = (f(x+step,data_for_subj,f_obs,prior,beta_prior) - f(x,data_for_subj,f_obs,prior,beta_prior))/step
    rough2ndderiv = (rough1stderiv2-rough1stderiv1)/step
    return rough2ndderiv
'''def get2ndDerivBasic(f,x,step):
    rough1stderiv1 = (f(x) - f(x-step))/step
    rough1stderiv2 = (f(x+step) - f(x))/step
    rough2ndderiv = (rough1stderiv2-rough1stderiv1)/step
    return rough2ndderiv
'''
def getHessian(f,x_vec,data_for_subj,f_obs,prior,beta_prior,step):
    element00=get2ndDerivMultiVar(f,x_vec,0,data_for_subj,f_obs,prior,beta_prior,step)
    element01=get1stDerivMultiVar(f,x_vec,0,data_for_subj,f_obs,prior,beta_prior,step)*get1stDerivMultiVar(f,x_vec,1,data_for_subj,f_obs,prior,beta_prior,step)
    element10=get1stDerivMultiVar(f,x_vec,1,data_for_subj,f_obs,prior,beta_prior,step)*get1stDerivMultiVar(f,x_vec,0,data_for_subj,f_obs,prior,beta_prior,step)
    element11=get2ndDerivMultiVar(f,x_vec,1,data_for_subj,f_obs,prior,beta_prior,step)
    return np.array([[element00,element01],[element10,element11]])
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


n = 80 # number of subjects #1500
n_iterations = 12
n_choices = 100 #2000

### I choose hyperparameters to use to generate the data ###

# true hyperparameters for alpha
alpha_mean = 0.5123456 # mean for alpha
alpha_stdev = 0.1123456 # std dev for alpha

# true hyperparameters for beta
beta_mean = 10.123456
beta_stdev = 0.1123456

# generate an alpha parameter for each subject
alphas = []
betas = []
for i in range(n):
    alpha=-1
    beta=-1
    while alpha<=0 or alpha>1: # make sure the generated value is never negative
        if alpha!=-1:
            print('this value of alpha of ',alpha,' is a possible problem')
        alpha = np.random.normal(loc=alpha_mean,scale=alpha_stdev)
    alphas.append(alpha)
    while beta<=0: # make sure the generated value is never negative
        if beta!=-1:
            print('this value of beta of ',beta,' is a possible problem')
        beta = np.random.normal(loc=beta_mean,scale=beta_stdev)
    betas.append(beta)

# stack the alphas and the betas together like (alpha1,beta1),(alpha2,beta2),...
true_params=np.stack((alphas,betas),axis=-1) 

#### Initial estimates for hyperparameters ####
#alpha_mean_est   = 0.7  # mean est for alpha
#alpha_stdev_est = 1 # std dev est for alpha

starting_val_mean_beta = 10
starting_val_stdev_beta = 0.1

starting_values_mean_alpha=[0.4]#np.arange(0.10,0.53,0.1)
starting_values_stdev_alpha=[0.1]#np.arange(0.1,0.12,0.01)

#starting_val_pairs=[[[]]] # [[[alpha1mean,alpha1stdev],[beta1mean,beta1stdev]],[[alpha2mean,alpha2stdev],[beta2mean,beta2stdev]],...]

'''stepsizes=np.linspace(0.000001,10,10)
rough2nds = np.zeros(len(stepsizes))
for ii,size in enumerate(stepsizes):
    rough2nds[ii] = get2ndDerivBasic(f,10,size)
plt.figure()
plt.plot(stepsizes,rough2nds)
plt.show()
    
'''

all_data = gen_data(true_params,n_choices,n) # generate the data using the true parameters for each subject


#### Do expectation maximisation ####
log_lik_max_prev=1000000000 # make really big so that first log lik is defs smalle
for starting_val_stdev_alpha in starting_values_stdev_alpha: # run code for multiple starting values
    for starting_val_mean_alpha in starting_values_mean_alpha:
        alpha_prior_est = [starting_val_mean_alpha,starting_val_stdev_alpha] # mean, std dev
        beta_prior_est=[starting_val_mean_beta,starting_val_stdev_beta]
        for k in range(n_iterations):
            m=[] # form of m: ((alpha_subj_1,beta_subj_1),(alpha_subj_2,beta_subj_2),...)
            Sigma=[] 
            total_log_lik = 0
            ##plt.figure()
            print('We are on iteration ',k)
            for subj in range(n):
                data_for_subj = all_data[subj]
                bnds = [(0.1,0.9123),(10,13)] # check if this is right 
                
                init_ests = [0.5,4] # initial estimates for the parameters we're trying to find 
                ### minimize the negative log posterior ###
                argmax_est = minimize(neg_log_post,init_ests, args=(data_for_subj,RW_obs,alpha_prior_est,beta_prior_est),bounds=bnds)
                
                #print("The true alpha is ",alphas[subj],". The re-generated alpha={0}".format(argmax_est.x[0]),'\n')
                m.append(argmax_est.x)
                total_log_lik += neg_log_post(m[-1],data_for_subj,RW_obs,alpha_prior_est,beta_prior_est)
                '''stepsizes=np.linspace(1e-6,1e-4,10)
                rough2nds = np.zeros(len(stepsizes))
                for ii,size in enumerate(stepsizes):
                    rough2nds[ii] = get2ndDeriv(neg_log_post,m[-1],data_for_subj,RW_obs,alpha_prior_est,size)
                plt.figure()
                plt.plot(stepsizes,rough2nds)
                plt.show()'''
                # test derivative code on known function
                #print('m[-1]',m[-1])
                #rough2ndderiv = get2ndDeriv(neg_log_post,m[-1],data_for_subj,RW_obs,alpha_prior_est,beta_prior_est,1e-4)

                hess = np.array([[-1,0],[0,-1]])#getHessian(log_post,m[-1],data_for_subj,RW_obs,alpha_prior_est,beta_prior_est,1e-4)#nd.Hessian(log_post)(m[-1],data_for_subj,RW_obs,alpha_prior_est,beta_prior_est)
                # take the hessian of log_post and not neg_log_post
                #print(hess)
                
                #det_hess = np.linalg.det(hess)
                #c = det_hess
                neg_inv_hess=-np.linalg.inv(hess)
                #print('neginvhess',neg_inv_hess)
                Sigma.append(neg_inv_hess)
            

                
                #Sigma.append(1/rough2ndderiv)
                
                
            
            
            #bounds=Bounds([0.5,1.5],[1,2.5]) # set upper and lower bounds for the parameters we're trying to estimate. [lower_p1,lower_p2],[upper_p1,upper_p2]
            
            
            #############################
            #init_ests = [0.75,2.1] # initial estimates for the parameters we're trying to find 
            #argmax_est = minimize(neglogpost,init_ests, args=(rows,RW_obs,alpha_prior_est,rho_prior_est),bounds=bounds)
            ### now we do the global stuff, the stuff that combines the info for individual subjects    
            m=np.array(m)
            #print('m',m)
            
            Sigma=np.array(Sigma)
            #print('sigma',Sigma)
            mu = np.average(m,axis=0)
            #print('mu',mu)
            Sigma_alpha=Sigma[:,0,0]
            Sigma_beta=Sigma[:,1,1]
            #print('Sigma alpha',Sigma_alpha)
            #print('Sigma beta',Sigma_beta)
            # The purpose of the tile and transpose functions below is to make sigma the same shape as m. For a given subject, sigma is the same for all the parameters. (todo: double check that) 
            #print('thingwe sqrt',1/n*np.sum((m[:,0]**2+Sigma_alpha),axis=0)- mu[0]**2)
            nu_alpha = np.sqrt(1/n*np.sum((m[:,0]**2+Sigma_alpha),axis=0)- mu[0]**2)
            nu_beta = np.sqrt(1/n*np.sum((m[:,1]**2+Sigma_beta),axis=0)- mu[1]**2)
            
            # mu_arr: [mean of the prior gaussian for alpha,mean of the prior gaussian for rho]
            # nu_arr: [std dev of the prior gaussian for alpha,std dev of the prior gaussian for rho]
            # these means and std devs are now our new estimates for the hyperparameters
            alpha_prior_est = [mu[0],nu_alpha]
            beta_prior_est = [mu[1],nu_beta]

            print('the total log lik for iteration ',k,' is ',total_log_lik)
            
            print('mean for alpha \t= ',mu[0])
            print('std dev for alpha \t=',nu_alpha)
            print('mean for beta \t= ',mu[1])
            print('std dev for beta \t=',nu_beta)

        #if total_log_lik<log_lik_max_prev:
        print('for alpha:')
        print('for starting value mean=',starting_val_mean_alpha,' and stdev=',starting_val_stdev_alpha)
        print('mean for alpha \t= ',mu[0])
        print('std dev for alpha \t=',nu_alpha)

        print('for beta:')
        print('for starting value mean=',starting_val_mean_beta,' and stdev=',starting_val_stdev_beta)
        print('mean for beta \t= ',mu[1])
        print('std dev for beta \t=',nu_beta)

        print()
        print('total log lik',total_log_lik)
        #log_lik_max_prev=total_log_lik
        

'''  
mean_range=np.arange(0.4,0.6,0.04)
scale_range=np.arange(0.01,0.2,0.04)
alpha_prior_range=[mean_range,scale_range]
z=np.zeros((len(mean_range),len(scale_range)))
for loci,loc in enumerate(alpha_prior_range[0]):
    for scalei,scale in enumerate(alpha_prior_range[1]):
        z[loci,scalei]=get_int_product(all_data,RW_obs,[loc,scale])
        print("int product for loc=",loc,"and scale=",scale,"is ",z[loci,scalei])

plot_this_in_3d(scale_range,loc_range,z,xlabel='stddev',ylabel='mean')
plt.show()
'''
plt.show()