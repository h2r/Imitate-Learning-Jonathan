from __future__ import print_function, division
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from skimage import io, transform
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		# Note that all of the layers have valid padding
		self.layer1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2)
		self.layer1_depth = nn.Conv2d(1, 16, kernel_size=7, stride=2)
		self.layer2 = nn.Conv2d(80, 32, kernel_size=1)
		self.layer3 = nn.Conv2d(32, 32, kernel_size=3)
		self.layer4 = nn.Conv2d(32, 32, kernel_size=3)
		self.spatial_softmax = SpatialSoftmax(53, 73, 32)
		self.fl1 = nn.Linear(64, 50)
		# This is where the concatenation of the output from spatialsoftmax
		# and the end effector information of the last 5 movements
		# are added. We use 7 points(x, y, z, a, b, c, d) so 50 + 35 = 85
		self.fl2 = nn.Linear(85, 50)
		# We use 7 instead of 6 in the paper for (linear velocity, angular velocity, grip {0, 1})
		# For today use 6 to simplify loss
		self.output = nn.Linear(50, 6)

	def forward(self, input_layer):
		rgb = input_layer[0]
		depth = input_layer[1]
		#7x5 quaternion + position 35 + 50 = 85 
		eof = input_layer[2]
		x_rgb = self.layer1_rgb(rgb)
		x_depth = self.layer1_depth(depth)
		x = F.relu(torch.cat([x_rgb, x_depth], dim=1))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		x = F.relu(self.layer4(x))
		x = self.spatial_softmax(x)
		x = F.relu(self.fl1(x))
		x = F.relu(self.fl2(torch.cat([x,eof], dim=1)))
		x = self.output(x)
		return x
		

class SpatialSoftmax(nn.Module):
	"""
	Spatial Softmax Implementation
	"""
	def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
		super(SpatialSoftmax, self).__init__()
		self.data_format = data_format
		self.height = height
		self.width = width
		self.channel = channel

		if temperature:
			self.temperature = Parameter(torch.ones(1)*temperature)
		else:
			self.temperature = 1.

		pos_x, pos_y = np.meshgrid(
			np.linspace(-1., 1., self.height),
			np.linspace(-1., 1., self.width))
		pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
		pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
		self.register_buffer('pos_x', pos_x)
		self.register_buffer('pos_y', pos_y)

	def forward(self, feature):
		if self.data_format == 'NHWC':
			feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
		else:
			feature = feature.view(-1, self.height*self.width)

		softmax_attention = F.softmax(feature/self.temperature, dim=-1)
		expected_x = torch.sum(Variable(self.pos_x)*softmax_attention, dim=1, keepdim=True)
		expected_y = torch.sum(Variable(self.pos_y)*softmax_attention, dim=1, keepdim=True)
		expected_xy = torch.cat([expected_x, expected_y], 1)
		feature_keypoints = expected_xy.view(-1, self.channel*2)

		return feature_keypoints

class BehaviorCloneLoss(nn.Module):
	"""
	The Loss function described in the paper
	"""
	def __init__(self, lamb_l2=0.01, lamb_l1=1.0, lamb_c=0.005, lamb_g=0.01):
		super(BehaviorCloneLoss, self).__init__()
		self.lamb_l2 = lamb_l2
		self.lamb_l1 = lamb_l1
		self.lamb_c = lamb_c
		self.lamb_g = lamb_g
		self.l2 = nn.MSELoss()
		self.l1 = nn.L1Loss()
		self.g = nn.CrossEntropyLoss()

	def forward(self, output, target):
		l2_loss = self.l2(output, target)
		l1_loss = self.l1(output, target)
		# For the arccos loss
		bs, n = output.shape
		num = torch.bmm(target.view(bs,1,n), output.view(bs,n,1))
		den = torch.bmm(torch.norm(target.view(bs,n,1),p=2,dim=1,keepdim=True),torch.norm(output.view(bs,n,1),p=2,dim=1,keepdim=True))
		c_loss = torch.mean(torch.acos(num/den))
		return self.lamb_l2*l2_loss + self.lamb_l1*l1_loss + self.lamb_c*c_loss

class BaxterDataset(Dataset):
	def __init__(self, root_dir, mode, prediction=False, folder=None):
		"""
		Args:
		    root_dir (string): Directory with all the images.
		    mode (string): either Train or Test for the mode
		"""
		self.sub_dirs = [x[0] for x in os.walk(root_dir)][1:]
		self.data = {"rgb": [], "depth": [], "eof": [], "labels": [], "gripper": []}
		counter = 1
		for sub_dir in self.sub_dirs:
			if !prediction:
				for root, _, file in os.walk(sub_dir):
					file = sorted(file)
					vectors = pd.read_csv(root+"/"+file[-1], header=-1)
					for i in range(10, len(file)-1, 2):
						if !prediction:
							if mode == "train" and counter > 2 and counter <= 10:
								prevs = [float(file[i-j][:-10]) for j in range(10, 1, -2)]
								eof = []
								for prev in prevs:
									eof.append([float(vectors[vectors[0]==prev][j]) for j in range(1,8)])
								label = [float(vectors[vectors[0]==float(file[i][:-10])][j]) for j in range(8,14)]
								gripper = float(vectors[vectors[0]==float(file[i][:-10])][14])
								depth_image = cv2.imread(root+"/"+file[i], 0)
								rgb_image = cv2.imread(root+"/"+file[i+1], -1)
								depth_image = cv2.resize(depth_image, (160,120))
								rgb_image = cv2.resize(rgb_image, (160,120))
								depth_image = np.reshape(depth_image, (1, depth_image.shape[0], -1))
								rgb_image = np.reshape(rgb_image, (3, rgb_image.shape[0], -1))
								self.data["depth"].append(depth_image)
								self.data["rgb"].append(rgb_image)
								self.data["eof"].append(np.array(eof).flatten())
								self.data["labels"].append(label)
								self.data["gripper"].append(gripper)
							if mode == "test" and counter > 0 and counter <= 2:
								prevs = [float(file[i-j][:-10]) for j in range(10, 1, -2)]
								eof = []
								for prev in prevs:
									eof.append([float(vectors[vectors[0]==prev][j]) for j in range(1,8)])
								label = [float(vectors[vectors[0]==float(file[i][:-10])][j]) for j in range(8,14)]
								gripper = float(vectors[vectors[0]==float(file[i][:-10])][14])
								depth_image = cv2.imread(root+"/"+file[i], 0)
								rgb_image = cv2.imread(root+"/"+file[i+1], -1)
								depth_image = cv2.resize(depth_image, (160,120))
								rgb_image = cv2.resize(rgb_image, (160,120))
								depth_image = np.reshape(depth_image, (1, depth_image.shape[0], -1))
								rgb_image = np.reshape(rgb_image, (3, rgb_image.shape[0], -1))
								self.data["depth"].append(depth_image)
								self.data["rgb"].append(rgb_image)
								self.data["eof"].append(np.array(eof).flatten())
								self.data["labels"].append(label)
								self.data["gripper"].append(gripper)
							if counter == 10:
								counter = 1
							else:
								counter += 1
				else:
					if sub_dir == folder:
						prevs = [float(file[i-j][:-10]) for j in range(10, 1, -2)]
						eof = []
						for prev in prevs:
							eof.append([float(vectors[vectors[0]==prev][j]) for j in range(1,8)])
						label = [float(vectors[vectors[0]==float(file[i][:-10])][j]) for j in range(8,14)]
						gripper = float(vectors[vectors[0]==float(file[i][:-10])][14])
						depth_image = cv2.imread(root+"/"+file[i], 0)
						rgb_image = cv2.imread(root+"/"+file[i+1], -1)
						depth_image = cv2.resize(depth_image, (160,120))
						rgb_image = cv2.resize(rgb_image, (160,120))
						depth_image = np.reshape(depth_image, (1, depth_image.shape[0], -1))
						rgb_image = np.reshape(rgb_image, (3, rgb_image.shape[0], -1))
						self.data["depth"].append(depth_image)
						self.data["rgb"].append(rgb_image)
						self.data["eof"].append(np.array(eof).flatten())
						self.data["labels"].append(label)
						self.data["gripper"].append(gripper)
		self.data["depth"] = Variable(torch.from_numpy(np.array(self.data["depth"])).type(torch.FloatTensor))
		self.data["rgb"] = Variable(torch.from_numpy(np.array(self.data["rgb"])).type(torch.FloatTensor))
		self.data["eof"] = Variable(torch.from_numpy(np.array(self.data["eof"])).type(torch.FloatTensor))
		self.data["labels"] = Variable(torch.from_numpy(np.array(self.data["labels"])).type(torch.FloatTensor))
		self.data["gripper"] = Variable(torch.from_numpy(np.array(self.data["gripper"])).type(torch.FloatTensor))

	def __len__(self):
		return self.data["labels"].shape[0]

	def __getitem__(self, idx):
		return (self.data["rgb"][idx], self.data["depth"][idx], self.data["eof"][idx], self.data["gripper"][idx]), self.data["labels"][idx]

def isCorrect(output, target, epsilon=0.1):
	return torch.sum(torch.sum(torch.abs(output-target), dim=1) < epsilon)

def train(root_dir, num_epochs=1000, bs=64, lr=0.009):
	modes = ["train", "test"]
	costs = {mode: [] for mode in modes}
	#accuracy = {mode: [] for mode in modes}
	model = Model()
	model.cuda()
	criterion = BehaviorCloneLoss().cuda()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	datasets = {mode: BaxterDataset(root_dir, mode) for mode in modes}
	dataloaders = {mode: DataLoader(datasets[mode], batch_size=bs, shuffle=True, num_workers=8) for mode in modes}
	data_sizes = {mode: len(datasets[mode]) for mode in modes}
	for epoch in range(num_epochs):
		if epoch % 10 == 0:
			print("-"*10)
			print("Epoch {}/{}".format(epoch, num_epochs))
			print("-"*10)
		for mode in modes:
			running_loss = 0.0
			running_correct = 0
			for input_layer, target in dataloaders[mode]:
				input_layer = [x.cuda() for x in input_layer]
				target = target.cuda()
				optimizer.zero_grad()
				output = model(input_layer)
				loss = criterion(output, target)
				loss.backward()
				optimizer.step()
				running_loss += loss
				#running_correct += isCorrect(output, target)
			# Print the cost and accuracy every 10 epoch
			if epoch % 10 == 0:
				cost = running_loss/data_sizes[mode]
				#acc = float(running_correct)/data_sizes[mode]
				print("{} Loss: {}".format(mode, cost))
				#print("{} Accuracy: {}".format(mode, acc))
				costs[mode].append(cost)
				#accuracy[mode].append(acc)
	# plot the cost and accuracy
	# _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	x = np.arange(0, len(costs["train"]))
	# ax1.plot(x, np.squeeze(costs["train"]), x, np.squeeze(costs["test"]), label=["Train", "Test"])
	# ax1.set_ylabel("Loss")
	# ax2.plot(x, np.squeeze(accuracy["train"]), x, np.squeeze(accuracy["test"]), label=["Train", "Test"])
	# ax2.set_ylabel('Accuracy')
	# ax2.set_xlabel('Iterations (per tens)')
	# ax1.set_title("Learning rate =" + str(lr))
	# plt.show()
	plt.plot(x, np.squeeze["train"], x, np.squeeze(costs["test"]))
	plt.ylabel("Loss")
	plt.xlabel("Iteration (per tens)")
	plt.title("Learning rate =" + str(lr))
	plt.legend(["Training", "Testing"])
	plt.show()
	torch.save(model.state_dict(), "torch_imp")

def predict(root_dir, sub_dir):
	dataset = BaxterDataset(root_dir, prediction=True, folder=sub_dir)
	model = Model()
	model.load_state_dict(torch.load("torch_imp"))
	model.eval()
	model.cuda()
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
	predictions = []
	for input_layer, _ in dataloader:
		input_layer = [x.cuda() for x in input_layer]
		output = model(input_layer)
		predictions.append(output)
	return predictions


if __name__ == '__main__':
	root_dir = "mini_test_data"
	sub_dir = "1524149383.29"
	train(root_dir)
	predict(root_dir, sub_dir)

