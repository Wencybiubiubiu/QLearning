import scipy.optimize as opt
import scipy.stats as sta
import numpy as np
import random
import pandas as pd 
import matplotlib.pyplot as plt
#initial value of rho
use_converge = 0
use_iter = 1
iteration = 15
scale = 100
ro = 0.9
converge_condition = 50
graph_scale = 0.1
#initial value of qn and qn+1
basic_q = []
for i in range(1,scale):
	basic_q.append(float(i)/scale)
q_array = [basic_q,basic_q]
next_q_array = [basic_q,basic_q]
print(q_array)

#initial value of reward array: normal distribution between (0,1)
lower, upper = 0, 1
mu, sigma = 0, 1
X = sta.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
R = X.rvs((2,scale-1))
#print(R)

def find_nearest(input_array,value,x_row):
	temp = []
	input_array = np.array(input_array)
	for i in range(len(input_array[x_row])):
		temp.append(np.abs(input_array[x_row][i]-value))
	temp = np.array(temp)
	idx = temp.argmin()
	return idx

def q(a,b):
	return sta.norm.cdf(b) - sta.norm.cdf(a)

def get_alpha_beta(q_value):
	gamma = 1/2 + np.log(q_value/(1-q_value))
	lower, upper = gamma-0.5, gamma
	mu, sigma = 0, 1
	X = sta.truncnorm(
	    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
	alpha = X.rvs(1)
	lower, upper = gamma, gamma+0.5
	mu, sigma = 0, 1
	Y = sta.truncnorm(
	    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
	beta = Y.rvs(1)
	return [alpha,beta,gamma]

# alpha = x[0]
# beta = x[1]
# q0n_plus_1 = x[2]
# q1n_plus_1 = x[3]
def f_0(x,qn0):
    #return -( (sta.norm.cdf(x[0])) + ro*q(x[0],1/2 + np.log(x[2]/(1-x[2])))*r_0_n_1 + ro*q(1/2 + np.log(x[2]/(1-x[2])),x[1])*r_1_n_1)
    return -( (sta.norm.cdf(x[0])) + ro*q(x[0],1/2 + np.log(qn0/(1-qn0)))*R[0][find_nearest(q_array,x[2],0)] + ro*q(1/2 + np.log(qn0/(1-qn0)),x[1])*R[1][find_nearest(q_array,x[3],1)])

def f_1(x,qn1):
    return -( (1-sta.norm.cdf(x[1])) + ro*q(x[0],1/2 + np.log(qn1/(1-qn1)))*R[0][find_nearest(q_array,x[2],0)] + ro*q(1/2 + np.log(qn1/(1-qn1)),x[1])*R[1][find_nearest(q_array,x[3],1)])

def optimize_one_iter_0(qn0):
	gamma = 1/2 + np.log(qn0/(1-qn0))
	cons = ({'type': 'eq',
	         'fun' : lambda x: np.array([
	         	x[0]-(gamma)+np.log(R[1][find_nearest(q_array,x[3],1)]/(1-R[0][find_nearest(q_array,x[2],0)])), 
	         	x[1]-(gamma)-np.log(R[0][find_nearest(q_array,x[2],0)]/(1-R[1][find_nearest(q_array,x[3],1)])),
	         	x[2]/(1-x[2])-qn0*float(q(x[0],gamma)/((1-qn0)*q(x[0]-1,gamma-1))),
	         	x[3]/(1-x[3])-qn0*float(q(gamma,x[1])/((1-qn0)*q(gamma-1,x[1]-1)))
	         	])})
	#print(random.randint(0,9))
	#x0 = np.array([2,2,q_array[0][random.randint(0,8)],q_array[1][random.randint(0,8)]])
	x_2 = float(q_array[0][random.randint(0,8)])
	x_3 = float(q_array[1][random.randint(0,8)])
	alpha_beta = get_alpha_beta(qn0)
	x0 = np.array([alpha_beta[0],alpha_beta[1],x_2,x_3,alpha_beta[2]])
	cx = opt.minimize(f_0, x0,qn0,constraints=cons)
	'''
	print('current iteration(qn0)',qn0)
	print('initial params',x0)
	print('final optimum',cx.fun)
	print('final params',cx.x)
	'''
	return cx

def optimize_one_iter_1(qn1):
	gamma = 1/2 + np.log(qn1/(1-qn1))
	cons = ({'type': 'eq',
	         'fun' : lambda x: np.array([
	         	x[0]-(gamma)+np.log(R[1][find_nearest(q_array,x[3],1)]/(1-R[0][find_nearest(q_array,x[2],0)])), 
	         	x[1]-(gamma)-np.log(R[0][find_nearest(q_array,x[2],0)]/(1-R[1][find_nearest(q_array,x[3],1)])),
	         	x[2]/(1-x[2])-qn1*float(q(x[0],gamma)/((1-qn1)*q(x[0]-1,gamma-1))),
	         	x[3]/(1-x[3])-qn1*float(q(gamma,x[1])/((1-qn1)*q(gamma-1,x[1]-1)))
	         	])})
	#print(random.randint(0,9))
	#x0 = np.array([2,2,q_array[0][random.randint(0,8)],q_array[1][random.randint(0,8)]])
	#x0 = np.array([2,2,q_array[0][random.randint(0,8)],q_array[1][random.randint(0,8)]])
	x_2 = float(q_array[0][random.randint(0,8)])
	x_3 = float(q_array[1][random.randint(0,8)])
	alpha_beta = get_alpha_beta(qn1)
	x0 = np.array([alpha_beta[0],alpha_beta[1],x_2,x_3,alpha_beta[2]])
	cx = opt.minimize(f_1, x0, qn1, constraints=cons)
	"""
	print('current iteration(qn1)',qn1)
	print('initial params',x0)
	print('final optimum',cx.fun)
	print('final params',cx.x)
	"""
	return cx

converge = 0
optimize_params0 = []
optimize_params1 = []

temp_opt_0 = []
for i in range(len(q_array[0])):
	temp_opt_0.append([['alpha','beta','q0_n_plus_1','q1_n_plus_1','gamma'],'reward'])
optimize_params0.append(temp_opt_0)

temp_opt_1 = []
for i in range(len(q_array[1])):
	temp_opt_1.append([['alpha','beta','q0_n_plus_1','q1_n_plus_1','gamma'],'reward'])
optimize_params1.append(temp_opt_1)

if use_converge:
	iter_num = 0
	while not converge:
		iter_num = iter_num + 1
		temp0 = []
		next_step0 = []
		for i in range(len(q_array[0])):
			opt_0 = optimize_one_iter_0(q_array[0][i])
			temp0.append(-opt_0.fun)
			next_step0.append([np.ndarray.tolist(opt_0.x),-opt_0.fun])

		temp1 = []
		next_step1 = []
		for i in range(len(q_array[1])):
			opt_1 = optimize_one_iter_1(q_array[1][i])
			temp1.append(-opt_1.fun)
			next_step1.append([np.ndarray.tolist(opt_1.x),-opt_1.fun])
		diff = 0
		for i in range(len(R[0])):
			diff += abs(R[0][i] - temp0[i])
		for i in range(len(R[1])):
			diff += abs(R[1][i] - temp1[i])
		if diff <= converge_condition:
			converge = 1
			print('converge!!!!!!')
			print('diff',diff)
		R[0] = temp0
		R[1] = temp1
		print('iteration',iter_num)
		print(R)
		print('\n')
		print('next_step0',next_step0)
		print('\n')
		print('next_step1',next_step1)
		print('\n')
		optimize_params0.append(next_step0)
		optimize_params1.append(next_step1)

if use_iter:
	for k in range(iteration):
		temp0 = []
		next_step0 = []
		for i in range(len(q_array[0])):
			opt_0 = optimize_one_iter_0(q_array[0][i])
			temp0.append(-opt_0.fun)
			next_step0.append([np.ndarray.tolist(opt_0.x),-opt_0.fun])

		temp1 = []
		next_step1 = []
		for i in range(len(q_array[1])):
			opt_1 = optimize_one_iter_1(q_array[1][i])
			temp1.append(-opt_1.fun)
			next_step1.append([np.ndarray.tolist(opt_1.x),-opt_1.fun])
		R[0] = temp0
		R[1] = temp1
		print('iteration',k)
		print(R)
		print('\n')
		print('next_step0',next_step0)
		print('\n')
		print('next_step1',next_step1)
		print('\n')
		optimize_params0.append(next_step0)
		optimize_params1.append(next_step1)

"""
incorrect loops
for i in range(len(q_array[0])):
	temp0 = []
	temp1 = []
	next_step = [[],[]]
	for j in range(len(q_array[1])):
		opt_0 = optimize_one_iter_0(q_array[0][i])
		opt_1 = optimize_one_iter_1(q_array[1][j])
		temp0.append(-opt_0.fun)
		temp1.append(-opt_1.fun)
		next_step[0].append(opt_0.x)
		next_step[1].append(opt_1.x)
	temp0 = np.array(temp0)
	temp1 = np.array(temp1)
	R[0][i] = temp0[temp0.argmax()]
	R[1][i] = temp1[temp1.argmax()]
	next_q_array[0][i] = next_step[0][temp0.argmax()][2]
	next_q_array[1][i] = next_step[1][temp1.argmax()][3]
for i in range(len(next_q_array[0])):
	next_q_array[0][i] = q_array[0][find_nearest(q_array,next_q_array[0][i],0)]
for i in range(len(next_q_array[1])):
	next_q_array[1][i] = q_array[1][find_nearest(q_array,next_q_array[1][i],1)]
"""
np.printoptions(precision=3, suppress=True)
print('\n')
print('optimal reward value:')
print(R)
print('\n')
print('matching next q0 state:')
print(optimize_params0)
print('\n')
print('matching next q1 state:')
print(optimize_params1)
df = pd.DataFrame(optimize_params0)
df.to_csv("qn0.csv")
df = pd.DataFrame(optimize_params1)
df.to_csv("qn1.csv")

'''
do plotting

1.       alpha_n, beta_n, gamma_n, and q_n? (basically this should be 3 line graphs)

2.       q_{n+1}^0, q_{n+1}^1 vs q_n (2 line graphs)

3.       R_h vs q for both h=0, h=1
'''

print('\n')

print('\n')

print('\n')
alpha_n_h_0 = []
beta_n_h_0 = []
gamma_n_h_0 = []
q_nplus_1_0 = []
R_h_0 = []

alpha_n_h_1 = []
beta_n_h_1 = []
gamma_n_h_1 = []
q_nplus_1_1 = []
R_h_1 = []

for i in range(scale-1):
	alpha_n_h_0.append(optimize_params0[len(optimize_params0)-1][i][0][0])
	beta_n_h_0.append(optimize_params0[len(optimize_params0)-1][i][0][1])
	gamma_n_h_0.append(optimize_params0[len(optimize_params0)-1][i][0][4])
	q_nplus_1_0.append(optimize_params0[len(optimize_params0)-1][i][0][2])
	R_h_0.append(optimize_params0[len(optimize_params0)-1][i][1])

for i in range(scale-1):
	alpha_n_h_1.append(optimize_params1[len(optimize_params1)-1][i][0][0])
	beta_n_h_1.append(optimize_params1[len(optimize_params1)-1][i][0][1])
	gamma_n_h_1.append(optimize_params1[len(optimize_params1)-1][i][0][4])
	q_nplus_1_1.append(optimize_params1[len(optimize_params1)-1][i][0][2])
	R_h_1.append(optimize_params1[len(optimize_params1)-1][i][1])


fig = plt.figure()
#plt.subplot(3,2,1)
fig.add_subplot(1, 1, 1).set_xticks(np.arange(0, 1, graph_scale))
plt.plot(q_array[0], alpha_n_h_0)
plt.plot(q_array[0], beta_n_h_0)
plt.plot(q_array[0], gamma_n_h_0)
plt.legend(['alpha_n_h0', 'beta_n_h0', 'gamma_n_h0'], loc='upper left')
plt.xlabel('q_n for h = 0')
plt.ylabel('alpha,beta,gamma')
plt.savefig('alpha-beta-gamma-vs-qnh0.png')
plt.close(fig)

fig = plt.figure()
#plt.subplot(3,2,2)
fig.add_subplot(1, 1, 1).set_xticks(np.arange(0, 1, graph_scale))
plt.plot(q_array[1], alpha_n_h_1)
plt.plot(q_array[1], beta_n_h_1)
plt.plot(q_array[1], gamma_n_h_1)
plt.legend(['alpha_n_h1', 'beta_n_h1', 'gamma_n_h1'], loc='upper left')
plt.xlabel('q_n for h = 1')
plt.ylabel('alpha,beta,gamma')
plt.savefig('alpha-beta-gamma-vs-qnh1.png')
plt.close(fig)

fig = plt.figure()
#plt.subplot(3,2,3)
fig.add_subplot(1, 1, 1).set_xticks(np.arange(0, 1, graph_scale))
plt.plot(q_array[0], R_h_0)
plt.xlabel('q_n')
plt.ylabel('R_h0')
plt.savefig('Rh0-vs-qn0.png')
plt.close(fig)

fig = plt.figure()
#plt.subplot(3,2,4)
fig.add_subplot(1, 1, 1).set_xticks(np.arange(0, 1, graph_scale))
plt.plot(q_array[1], R_h_1)
plt.xlabel('q_n')
plt.ylabel('R_h1')
plt.savefig('Rh1-vs-qn1.png')
plt.close(fig)

fig = plt.figure()
#plt.subplot(3,2,5)
fig.add_subplot(1, 1, 1).set_xticks(np.arange(0, 1, graph_scale))
plt.plot(q_array[1], q_nplus_1_0)
plt.plot(q_array[1], q_nplus_1_1)
plt.legend(['q_{n+1}^0', 'q_{n+1}^1'], loc='upper left')
plt.xlabel('q_n')
plt.ylabel('q_{n+1}')
plt.savefig('qnplus1_0-qnplus1_1-vs-qn.png')
plt.close(fig)

fig = plt.figure()
#plt.subplot(3,2,5)
fig.add_subplot(1, 1, 1).set_xticks(np.arange(0, 1, graph_scale))
plt.plot(q_array[0], R_h_0)
plt.plot(q_array[0], R_h_1)
plt.legend(['R_h_0', 'R_h_1'], loc='upper left')
plt.xlabel('qn')
plt.ylabel('Reward')
plt.savefig('reward.png')
plt.close(fig)
#plt.axis([0, 6, 0, 20])
#plt.show()




