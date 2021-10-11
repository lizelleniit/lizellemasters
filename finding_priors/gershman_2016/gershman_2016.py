import numpy as np
import matplotlib.pyplot as plt

def update_V(V,r,c,eta):
	V[c] = V[c]+eta*(r-V[c])
	return V

def p_choices(V,beta): # find the probabilities of the actions given a value function
	p_choices_ar = np.zeros(n_options)
	for c in range(n_options):
		p_choices_ar[c] = 1/(1+np.exp(-beta*(V[c]-V[1-c])))
	return p_choices_ar

def choose(V,beta): # make a decision on which action to take based on the current value function
	p_choices_ar = p_choices(V,beta)
	rand = np.random.random()
	cumulative_probs = np.zeros(len(p_choices_ar))
	cumulative_probs[0]=p_choices_ar[0]
	if rand<=cumulative_probs[0]:
		return 0
	for i in range(1,len(p_choices_ar)):
		cumulative_probs[i]=cumulative_probs[i-1]+p_choices_ar[i]
		if cumulative_probs[i-1]<rand<=cumulative_probs[i]:
			return i 
	print("Something went wrong with the choose function.")
	return 999

def get_r(a):
	r_probs = np.array([0.2,0.8])
	p_r = r_probs[a]
	rand = np.random.random()
	if rand<p_r:
		return 1
	else:
		return 0

def gen_data_one_subj(n_choices,eta,beta):
	V=np.zeros(n_options)
	actions = np.zeros(n_choices,dtype=int)
	rewards = np.zeros(n_choices)
	for i in range(n_choices):
		actions[i] = choose(V,beta)
		rewards[i] = get_r(actions[i])
		V=update_V(V,rewards[i],actions[i],eta)
	return actions,rewards

def gen_data_n_subjects(n_subjects):
	# set mean and std dev for hyperparameters
	#eta_loc = 0.8 # mean
	#eta_scale = 0.1 # std dev
	etas = [0.1,0.2,0.1,0.3]
	all_data = []
	for subj in range(n_subjects):
		#eta = np.random.normal(loc=eta_loc,scale=eta_scale)
		#print('eta for subject ',subj,' is ',eta)
		actions,rewards = gen_data_one_subj(n_choices,etas[subj],beta)
		all_data.append([actions,rewards])
		#f = open('data\dummydata'+str(subj)+'.txt','w')
		#f.write("a\t")
		#f.write("r\n")
		#for i in range(len(actions)):
		#	f.write(str(actions[i]))
		#	f.write('\t')
		#	f.write(str(rewards[i]))
		#	f.write('\n')
		#f.close()
	return all_data

def get_log_lik(data_indiv,eta,beta):
	actions=data_indiv[0]
	rewards=data_indiv[1]
	V=np.zeros(n_options)
	log_lik = 0
	for i in range(len(actions)):
		log_lik += np.log(p_choices(V,beta)[actions[i]])

		V=update_V(V,rewards[i],actions[i],eta)
	return log_lik
beta=6
n_options = 2
n_choices = 100
all_data = gen_data_n_subjects(4)


eta_range = np.arange(0.1,1,0.001)
log_lik_ar = np.zeros(len(eta_range))
for etai,eta in enumerate(eta_range):
	log_lik_ar[etai] = get_log_lik(all_data[0],eta,beta)


plt.figure()
plt.plot(eta_range,log_lik_ar)
plt.show()

