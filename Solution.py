"""
@project = 2019101706
@file = solution
@author = MK
@create_time = 2019/10/18 2:07
"""

import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import neighbors

TIME_1 = 0.1  # Time
TIME_2 = 0.5
TIME_3 = 1
SIZE_1 = 5  # Size
SIZE_2 = 30
SIZE_3 = 100
F_1 = 2  # f: Ability
F_2 = 5  # should be chosen by q-learning
F_3 = 20
R_1 = 1.0
R_2 = 4.0  # should be chosen by q-learning
R_3 = 2.0
RMAX = 3  # the range, will be changed in Env_1
RMIN = 0.5
EPISODE_NUM = 100  # the number of the q-learning episodes
MASTER_NUM = 100  # the number of the whole process
PIC = 5
PATH = 'train_set.txt'

task_set = []  # the set of tasks
cost_set = [0, 0, 0, 0]  # the cost of each platform
func2_set = []  # the set of tasks for func2
platform_set = []  # the set of the chosen platform

class Task:
	def __init__(self, a_size=10, b_size=10, d_num=0.01, t_tola=0.1, platform_num=1):
		self.a_size = a_size  # output data size: bit(1 ~ 10)
		self.b_size = b_size  # input data size: bit(10 ~ 100)
		self.d_num = d_num  # occupyed CPU num per bit: (0.01 ~ 0.1)
		self.t_tola = t_tola  # time can be tolerant: (0.1 ~ 10)
		self.platform_num = platform_num  # the chosen platform: (1, 2, 3)

class select_knn:
	knn = neighbors.KNeighborsClassifier()  # 取得knn分类器
	train_data = ''

	def __init__(self):
		self.train_data = []  # 初始化训练样本
		return

	def get_train_data(self, path):  # 文件操作，读取训练样本
		f = open(path, 'r')
		line_list = f.readlines()
		f.close()
		train_set = []
		lables = []
		for i in range(0, len(line_list)):
			line = line_list[i].strip('\n')
			line = line.split(',')
			a = float(line[0])
			b = float(line[1])
			c = float(line[2])
			train_set.append([a, b])
			lables.append(c)
		self.train_data = [np.array(train_set), np.array(lables)]  # data对应着时延和任务大小

	def train(self):
		self.knn.fit(self.train_data[0], self.train_data[1])  # 导入数据进行训练

	def predict(self, task_time, task_cpusize):  # 对预测数据集进行卸载位置预测
		task_point = [[task_time, task_cpusize]]
		final_result = int(self.knn.predict(task_point))
		return final_result

class QLearningTable:
	def __init__(self, network, states, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.Action = []
		self.states = 0
		self.datasize = states
		if network == 1:
			self.states = np.power(RMAX, states)
			self.fill_table(self.states, network)
		else:
			self.states = np.power(RMAX+1, states)
			self.fill_table(self.states, network)
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = np.zeros([self.states, 1])

	def choose_action(self, observation):
		return self.Action[observation]

	def learn(self, s, a, r, s_):
		q_predict = self.q_table[s, 0]
		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table[s_, :].max()  # next state is not terminal
		else:
			q_target = r  # next state is terminal
		self.q_table[s, 0] += self.lr * (q_target - q_predict)  # update

	def fill_table(self, num, n_tp):
		if n_tp == 1:
			for i in range(num):
				tmp_a = np.ones(self.datasize, dtype=np.int64)
				# print(self.datasize)
				j = self.datasize - 1
				tmp_e = i
				if tmp_e == 1:
					units = tmp_e % RMAX
					tmp_a[j] += units
					tens = tmp_e / RMAX
					tmp_a[j - 1] += tens
				while tmp_e != 1:
					units = tmp_e % RMAX
					tmp_a[j] += units
					tens = tmp_e / RMAX
					if tens < RMAX:
						tmp_a[j - 1] += tens
						break
					else:
						tmp_e /= RMAX
						tmp_e = math.floor(tmp_e)
						j -= 1
				self.Action.insert(i, tmp_a)
		else:
			for i in range(num):
				for i in range(num):
					tmp_a = np.zeros(self.datasize, dtype=np.int64)
					j = self.datasize - 1
					tmp_e = i
					if tmp_e == 1:
						units = tmp_e % (RMAX+1)
						tmp_a[j] += units
						tens = tmp_e / (RMAX+1)
						tmp_a[j - 1] += tens
					while tmp_e != 1:
						units = tmp_e % (RMAX+1)
						tmp_a[j] += units
						tens = tmp_e / (RMAX+1)
						if tens < (RMAX+1):
							tmp_a[j - 1] += tens
							break
						else:
							tmp_e /= (RMAX+1)
							tmp_e = math.floor(tmp_e)
							j -= 1
					self.Action.insert(i, tmp_a)

class Env_1:
	def __init__(self):
		self.S = 0

	def reset(self, l):
		s = np.random.randint(0, np.power(RMAX, l))
		self.S = s
		return s

	def render(self, o):
		self.S = o

	def step(self, action):
		s_ = 1
		# terminal or not
		if s_ == 1:
			done = True
			s_ = 'terminal'
		else:
			done = False

		# reward function
		flag = False
		for i in range(len(action)):
			if action[i] < RMIN or action[i] > RMAX:
				flag = True
				break
		total_action = np.sum(action)
		if total_action > RMAX:
			reward = -1
		elif flag:
			reward = -1
		else:
			reward = 1 / total_action

		return s_, reward, done

class Env_2:
	def __init__(self, a):
		self.S = 0
		self.source_1 = a
		self.cost = 0

	def reset(self, l):
		s = np.random.randint(0, np.power(RMAX+1, l))
		self.S = s
		return s

	def render(self, o):
		self.S = o

	def step(self, action):
		s_ = 1
		# terminal or not
		if s_ == 1:
			done = True
			s_ = 'terminal'
		else:
			done = False

		# reward function
		reward = 0
		total_action = np.sum(action)
		if total_action > RMAX:
			reward = -1
			return s_, reward, done
		for i in range(len(action)):
			if action[i] == 0:
				continue
			r_2_tmp = self.source_1[i]*np.log2(120) / R_2
			left_tmp = func2_set[i].d_num*func2_set[i].b_size / action[i]
			tmp_r = func2_set[i].a_size/r_2_tmp
			right_tmp = func2_set[i].t_tola - tmp_r
			tmp_rr = func2_set[i].b_size/r_2_tmp
			right_tmp -= tmp_rr
			if 0 in action or left_tmp > right_tmp:
				reward_tmp = (1-action[i]/func2_set[i].d_num)*func2_set[i].b_size
				reward_tmp *= func2_set[i].d_num
				reward_tmp /= (func2_set[i].d_num-action[i])
				self.cost = tmp_r + tmp_rr + func2_set[i].b_size + reward_tmp
				reward += 1 / self.cost
			else:
				self.cost = tmp_r+tmp_rr+left_tmp
				reward += 1 / self.cost

		return s_, reward, done


# the function of creating the task
def create_task(idx):
	a_size_tmp = np.random.randint(1, 10, dtype=np.int64)
	b_size_tmp = np.random.randint(10, 100, dtype=np.int64)
	d_num_tmp = np.random.uniform(0.01, 0.1)
	t_tola_tmp = np.random.uniform(0.1, 10)
	task = Task(a_size_tmp, b_size_tmp, d_num_tmp, t_tola_tmp)
	# print(task.a_size)
	# print(task.b_size)
	# print(task.d_num)
	# print(task.t_tola)
	task_set.insert(idx, task)


# make sure the platform for handling the task
def func1_knn(n):
	tmp_platform = []
	sknn = select_knn()
	sknn.get_train_data(PATH)
	sknn.train()
	for i in range(n):
		t_tmp = task_set[i].t_tola
		d_tmp = task_set[i].d_num
		num = sknn.predict(t_tmp, d_tmp)
		tmp_platform.insert(i, num)
	return tmp_platform


# the process of q-learning iteration
def update(RL, env, l, flac, total_l):
	if flac == 1:
		print('n = ', total_l)
		print('Source 1 is training......')
	if flac == 2:
		print('Source 2 is training......')
	for episode in range(EPISODE_NUM):
		# initial observation
		observation = env.reset(l)
		step_counter = 0

		while True:

			# fresh env, move to state s_ in the previous step
			env.render(observation)

			# RL choose action based on observation
			action = RL.choose_action(observation)

			# RL take action and get next observation and reward
			observation_, reward, done = env.step(action)

			# RL learn from this transition
			RL.learn(observation, action, reward, str(observation_))

			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				break

		interaction = 'Episode %s: ' % (episode + 1)
		print('\r{}'.format(interaction), end='')

	# end of game
	if flac == 1:
		print('Source 1 is training Over')
		idx =  np.argmax(RL.q_table)
		# print(RL.Action[idx])
		return RL.Action[idx]
	if flac == 2:
		print('Source 2 is training Over')
		idx = np.argmax(RL.q_table)
		# print(RL.Action[idx])
		return RL.Action[idx]


# get the best A & B for set x
def func2_ql(l, total_l):
	c = np.zeros(l, dtype='int')  # c is the vector of A, source 1
	e = np.zeros(l, dtype='int')  # e is the vector of B, source 2
	if l == 0:
		return [c, e]
	# RMAX = int(input("Please set the W & U: "))
	# print(RMAX)
	RL_1 = QLearningTable(network=1, states=l)  # Create the Q-table
	RL_2 = QLearningTable(network=2, states=l)
	env_1 = Env_1()  # Create the environment
	c = update(RL_1, env_1, l, 1, total_l)  # The algorithm: Q-learning
	env_2 = Env_2(c)
	e = update(RL_2, env_2, l, 2, total_l)  # The algorithm: Q-learning
	return [c, e]


# plot the result
def plot_data():
	n = int(input("Please Input The Number of Tasks: "))
	x = np.linspace(1, n, n)
	y = []
	for i in range(1, n+1):
		y.insert(i-1, master(i))
	plt.figure()
	plt.title("The Cost Curve")
	plt.xlim(0, 6)
	plt.ylim(0, 30)
	plt.xlabel("the number of tasks")
	plt.ylabel("The Cost of Calculation")
	plt.scatter(x, y, s=30, color='red', alpha=1)  # s为size，按每个点的坐标绘制，alpha为透明度
	plt.show()


# calculate the total cost of 3 platforms
def calculate_cost():
	total_cost = np.sum(cost_set)
	return total_cost


# calculate the total cost of vector b
def calculate_cost_b(vt_b, vt_a):
	# cost function
	reward = 0
	for i in range(len(vt_b)):
		if vt_b[i] == 0:
			continue
		r_2_tmp = vt_a[i] * np.log2(120) / R_2
		left_tmp = func2_set[i].d_num * func2_set[i].b_size / vt_b[i]
		tmp_r = func2_set[i].a_size / r_2_tmp
		right_tmp = func2_set[i].t_tola - tmp_r
		tmp_rr = func2_set[i].b_size / r_2_tmp
		right_tmp -= tmp_rr
		if 0 in vt_b or left_tmp > right_tmp:
			reward_tmp = (1 - vt_b[i] / func2_set[i].d_num) * func2_set[i].b_size
			reward_tmp *= func2_set[i].d_num
			reward_tmp /= (func2_set[i].d_num - vt_b[i])
			reward_a = tmp_r + tmp_rr + func2_set[i].b_size + reward_tmp
			reward += 1 / reward_a
		else:
			reward_a = tmp_r + tmp_rr + left_tmp
			reward += 1 / reward_a

	return reward


# clear the sets
def clear_sets():
	del task_set[:]
	del cost_set[:]
	del func2_set[:]
	del platform_set[:]
	for i in range(4):
		cost_set.insert(i, 0)


# master, main interface
def master(n):
	total_num = 0
	for x in range(MASTER_NUM):
		t = 0
		for i in range(n):
			create_task(i)
		platform_set = func1_knn(n)
		# print(platform_set)
		for i in range(n):
			if platform_set[i] == 1:
				tmp = (task_set[i].b_size * task_set[i].d_num) / F_1
				tmp += task_set[i].a_size / R_1
				if tmp > task_set[i].t_tola:  # if t >
					platform_set[i] = 2  # choose the platform 2
				else:
					cost_set[1] += tmp  # add the task code
			if platform_set[i] == 2:
				func2_set.insert(t, task_set[i])  # put the task into set epsilon
				t += 1
			if platform_set[i] == 3:
				a = task_set[i].a_size
				b = task_set[i].b_size
				d = task_set[i].d_num
				cost = b / R_3 + 1 + (b * d) / F_3 + a / R_3
				cost_set[3] += cost
			task_set[i].platform_num = platform_set[i]

		length_func2 = len(func2_set)
		[p_a, p_b] = func2_ql(length_func2, n)  # get the best a b from q-learning
		cost_b = calculate_cost_b(p_b, p_a)
		cost_set[2] = np.sum(p_a) + cost_b
		total_num += calculate_cost()
		clear_sets()
	rounda = 'n = %s ' % n
	print('{}'.format(rounda), end='')
	average_num = total_num / MASTER_NUM
	result = 'total cost = %s ' % average_num
	print('\n{}\n\n'.format(result), end='')
	return average_num


# main function
if __name__ == "__main__":
	for i in range(PIC):
		plot_data()

