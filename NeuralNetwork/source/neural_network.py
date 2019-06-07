import numpy
import matplotlib.pyplot as plt
import scipy.special
import scipy.ndimage.interpolation
import json
import time
import progressbar
import matplotlib.animation as anim

# 神经网络类定义
class neuralNetwork:

	# 初始化神经网络
	def __init__(self,inputnodes,hiddennodes,hiddennodes_2,outputnodes,learningrate):
		# 设置神经网络的输入层、隐藏层、输出层、的结点数和学习率
		self.inodes = inputnodes
		self.hnodes = hiddennodes                # 第一隐藏层结点数
		self.hnodes_2 = hiddennodes_2            # 第二隐藏层结点数
		#self.hnodes_3 = hiddennodes_3            # 第三隐藏层结点数
		self.onodes = outputnodes
		# 学习率
		self.lr = learningrate
		
		# （常规版）链接权重矩阵,随机权重在-0.5至0.5之间（三层神经网络）
		self.wih = (numpy.random.rand(hiddennodes,inputnodes)-0.5)            
		self.who = (numpy.random.rand(outputnodes,hiddennodes)-0.5)           
		# （进阶版）链接权重矩阵,随机权重在-0.5至0.5之间（三层神经网络）
		self.wih_ = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(hiddennodes,inputnodes))          # 输入层到第一隐藏层权重矩阵
		self.wh12_ = numpy.random.normal(0.0,pow(self.hnodes_2,0.5),(hiddennodes_2,hiddennodes))     # 第一隐藏层到第二隐藏层权重矩阵
		#self.wh23_ = numpy.random.normal(0.0,pow(self.hnodes_3,0.5),(hiddennodes_3,hiddennodes_2))   # 第二隐藏层到第三隐藏层权重矩阵
		self.who_ = numpy.random.normal(0.0,pow(self.onodes,-0.5),(outputnodes,hiddennodes_2))         # 第三隐藏层到输出层权重矩阵

		#定义激活函数
		self.activation_function = lambda x : scipy.special.expit(x)


	# 训练神经网络
	def train(self,inputs_list,targets_list):
		# 将输入信号列表和目标信号列表转换成列向量
		inputs = numpy.array(inputs_list,ndmin=2).T
		targets = numpy.array(targets_list,ndmin=2).T

		# 第一隐藏层的输入信号：
		hidden_inputs = numpy.dot(self.wih_,inputs)
		# 第一隐藏层的输出信号（激活函数作用）：
		hidden_outputs = self.activation_function(hidden_inputs)

		# 第二隐藏层的输入信号:
		hidden_inputs_2 = numpy.dot(self.wh12_,hidden_outputs)
		# 第二层隐藏层的输出信号：
		hidden_outputs_2 = self.activation_function(hidden_inputs_2)
		'''
		# 第三隐藏层的输入信号：
		hidden_inputs_3 = numpy.dot(self.wh23_,hidden_outputs_2)
		# 第三隐藏层的输出信号：
		hidden_outputs_3 = self.activation_function(hidden_inputs_3)
		'''
		# 输出层的输入信号：
		final_inputs = numpy.dot(self.who_,hidden_outputs_2)
		# 输出层的输出信号：
		final_outputs = self.activation_function(final_inputs)

		# 计算输出层误差向量
		output_errors = targets - final_outputs
		# 计算第三隐藏层误差向量
		#hidden_errors_3 = numpy.dot(self.who_.T,output_errors)
		# 计算第二隐藏层的误差向量
		hidden_errors_2 = numpy.dot(self.who_.T,output_errors)
		# 计算第一隐藏层的误差向量
		hidden_errors = numpy.dot(self.wh12_.T,hidden_errors_2)

		''' 优化链接权重值 '''
		# 第三隐藏层与输出层间的链接权重优化
		#self.who_ += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs_3))
		# 第二隐藏层与第三隐藏层间的链接权重优化
		self.who_ += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs_2))
		# 第一隐藏层与第二隐藏层间的链接权重优化
		self.wh12_ += self.lr * numpy.dot((hidden_errors_2 * hidden_outputs_2 * (1.0 - hidden_outputs_2)),numpy.transpose(hidden_outputs))
		# 输入层与第一隐藏层间的链接权重优化
		self.wih_ += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
		
		
		#return self.query(inputs_list)

	# 查询
	def query(self,inputs_list):
		# 将输入列表转成numpy向量对象并转置为列向量
		inputs = numpy.array(inputs_list,ndmin=2).T
		# 第一隐藏层结点的输入信号：权重矩阵与输入信号向量的乘积
		self.hidden_inputs = numpy.dot(self.wih_,inputs)
		# 第一隐藏层结点的输出信号：经过S函数的加权求和值
		self.hidden_outputs = self.activation_function(self.hidden_inputs)

		# 第二隐藏层的输入信号:
		self.hidden_inputs_2 = numpy.dot(self.wh12_,self.hidden_outputs)
		# 第二层隐藏层的输出信号：
		self.hidden_outputs_2 = self.activation_function(self.hidden_inputs_2)
		'''
		# 第三隐藏层的输入信号：
		self.hidden_inputs_3 = numpy.dot(self.wh23_,self.hidden_outputs_2)
		# 第三隐藏层的输出信号：
		self.hidden_outputs_3 = self.activation_function(self.hidden_inputs_3)
		'''
		# 输出层结点的输入信号：
		self.final_inputs = numpy.dot(self.who_,self.hidden_outputs_2)
		# 输出层结点的最终输出信号：
		self.final_outputs = self.activation_function(self.final_inputs)

		# 返回最终输出信号
		return self.final_outputs
def test(Network,test_dataset_name):
	Network.wih_ = numpy.loadtxt('wih_file.csv')
	Network.wh12_ = numpy.loadtxt('wh12_file.csv')
	#Network.wh23_ = numpy.loadtxt('wh23_file.csv')
	Network.who_ = numpy.loadtxt('who_file.csv')

	# 准备测试数据
	test_data_file = open(test_dataset_name,'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()

	print('\n')
	print("Testing...\n")
	# 统计
	correct_test = 0
	all_test = 0
	correct = [0,0,0,0,0,0,0,0,0,0]
	num_counter = [0,0,0,0,0,0,0,0,0,0]

	#测试进度条
	p_test = progressbar.ProgressBar()
	p_test.start(len(test_data_list))

	# 动画显示
	#plt.figure(1)

	for imag_list in test_data_list:
		all_values = imag_list.split(',')
		lable = int(all_values[0])
		scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		imag_array = numpy.asfarray(scaled_input).reshape((28,28))
		'''
		plt.imshow(imag_array,cmap='Greys',animated=True)
		plt.draw()
		plt.pause(0.00001)
		'''
		net_answer = Network.query(scaled_input).tolist().index(max(Network.final_outputs))
		num_counter[lable] += 1
		
		if lable == int(net_answer):
			correct_test += 1
			correct[lable] += 1
		p_test.update(all_test + 1)
		all_test += 1

	p_test.finish()
	print("Finish Test.\n")

	# 网络性能
	performance = correct_test/all_test
	Per_num_performance = []
	for i in range(10):
		# 测试集可能不包含某些数字，故捕捉除以0异常
		try:
			Per_num_performance.append(correct[i]/num_counter[i])
		except ZeroDivisionError:
			Per_num_performance.append(0)

	print("The correctRate of per number： ",Per_num_performance)
	print("Performance of the NeuralNetwork： ",performance*100)
	return performance

# 定义网络规模与学习率
input_nodes = 784
hidden_nodes = 700
hidden_nodes_2 = 700
#hidden_nodes_3 = 100
output_nodes = 10
learningrate = 0.0001

if __name__ == "__main__":

	# 定义训练世代数
	epochs = 5

	#创建神经网络实例
	Net = neuralNetwork(input_nodes,hidden_nodes,hidden_nodes_2,output_nodes,learningrate)

	#plt.imshow(final_outputs,interpolation="nearest")

	# 准备训练数据
	data_file = open("mnist_train.csv",'r')
	data_list = data_file.readlines()
	N_train = len(data_list)
	data_file.close()
	# 动画显示
	#plt.figure(1)

	print("Training：", epochs, "epochs...")
	for e in range(epochs):
		# 训练进度条
		print('\nThe '+str(e+1)+'th epoch trainning:\n')
		p_train = progressbar.ProgressBar()
		p_train.start(N_train)
		i = 0

		for img_list in data_list:
			# 以逗号分割记录
			all_values = img_list.split(',')
			# 将0-255映射到0.01-0.99
			scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			imag_array = numpy.asfarray(scaled_input).reshape((28,28))
			#plt.imshow(imag_array,cmap='Greys',animated=True)
			#plt.draw()
			#plt.pause(0.00001)

			#旋转图像生成新的训练集
			input_plus_10imag = scipy.ndimage.interpolation.rotate(imag_array,10,cval=0.01,reshape=False)
			input_minus_10imag = scipy.ndimage.interpolation.rotate(imag_array,-10,cval=0.01,reshape=False)
			input_plus10 = input_plus_10imag.reshape((1,784))
			input_minus10 = input_minus_10imag.reshape((1, 784))
			# 根据标签创建目标值向量
			targets = numpy.zeros(output_nodes) + 0.01
			targets[int(all_values[0])] = 0.99

			# 用三个训练集训练神经网络
			Net.train(scaled_input,targets)
			Net.train(input_plus10,targets)
			Net.train(input_minus10,targets)

			#time.sleep(0.01)
			p_train.update(i+1)
			i+=1
		p_train.finish()
		
	print("\nTrainning finish.\n")

	# 将训练好的神经网络链接权重输出到csv文件中
	numpy.savetxt('wih_file.csv',Net.wih_,fmt='%f')
	numpy.savetxt('wh12_file.csv',Net.wh12_,fmt='%f')
	#numpy.savetxt('wh23_file.csv',Net.wh23_,fmt='%f')
	numpy.savetxt('who_file.csv',Net.who_,fmt='%f')

	