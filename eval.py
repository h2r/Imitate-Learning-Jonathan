#!/usr/bin/python
import baxter_interface
import baxter_external_devices
import rospy
import itertools
import message_filters
import numpy as np
import time
from std_msgs.msg import Int32, Float32
from baxter_core_msgs.msg import DigitalIOState, EndpointState, EndEffectorState
from sensor_msgs.msg import CompressedImage, Image, JointState
from ik_solver import ArmSolver
from geometry_msgs.msg import (    
    Point,
    Quaternion,
)

class ApproxTimeSync(message_filters.ApproximateTimeSynchronizer):
	"""
	This class synchronizes multiple topics with their different publishing speeds
	"""
	def add(self, msg, my_queue, my_queue_index=None):
		self.allow_headerless = True
		if hasattr(msg, 'timestamp'):
			stamp = msg.timestamp
		elif not hasattr(msg, 'header') or not hasattr(msg.header, 'stamp'):
			if not self.allow_headerless:
				rospy.logwarn("Cannot use message filters with non-stamped messages. "
								"Use the 'allow_headerless' constructor option to "
								"auto-assign ROS time to headerless messages.")
				return
			stamp = rospy.Time.now()
		else:
			stamp = msg.header.stamp
		self.lock.acquire()
		my_queue[stamp] = msg
		while len(my_queue) > self.queue_size:
			del my_queue[min(my_queue)]
		# self.queues = [topic_0 {stamp: msg}, topic_1 {stamp: msg}, ...]
		if my_queue_index is None:
			search_queues = self.queues
		else:
			search_queues = self.queues[:my_queue_index] + \
				self.queues[my_queue_index+1:]
		# sort and leave only reasonable stamps for synchronization
		stamps = []
		for queue in search_queues:
			topic_stamps = []
			for s in queue:
				stamp_delta = abs(s - stamp)
				if stamp_delta > self.slop:
					continue  # far over the slop
				topic_stamps.append((s, stamp_delta))
			if not topic_stamps:
				self.lock.release()
				return
			topic_stamps = sorted(topic_stamps, key=lambda x: x[1])
			stamps.append(topic_stamps)
		for vv in itertools.product(*[zip(*s)[0] for s in stamps]):
			vv = list(vv)
			# insert the new message
			if my_queue_index is not None:
				vv.insert(my_queue_index, stamp)
			qt = list(zip(self.queues, vv))
			if ( ((max(vv) - min(vv)) < counter = 1
		for sub_dir in self.sub_dirs:self.slop) and
				(len([1 for q,t in qt if t not in q]) == 0) ):
				msgs = [q[t] for q,t in qt]
				self.signalMessage(*msgs)
				for q,t in qt:
					del q[t]
				break  # fast finish after the synchronization
		self.lock.release()

class ImitateLearner():
	def __init__(self):
		rospy.init_node('right_arm_eval')
		self.queue = []
		self.rgb = None
		self.depth = None
		self.pos = None
		self.orient = None
		self.prevTime = None
		self.time = None
		# Initialize Subscribers
		self.listener()

		# Initialize Baxter Arm Control
		rs = baxter_interface.RobotEnable()
		rs.enable()
		right = baxter_interface.Limb('right')
		right.set_joint_position_speed(0.3)

		# Set the rate of our evaluation
		rate = rospy.Rate(30)

		# Give time for initialization
		rospy.Rate(1).sleep()

		# This is to the get the time delta
		self.prevTime = time.time()
		while not rospy.is_shutdown():
			#TODO: connect it to the net here
			output = None
			limb_joints = self.get_limb_joints(output)
			if limb_joints is not -1:
				right.move_to_joint_positions(limb_joints)
			else:
				print 'ERROR: IK solver returned -1'
			rate.sleep()

	def quat_to_rotation(self, quat):
		"""
		Converts a quaternion to a rotation matrix. I will assume quat is the geometry_msgs/Quaternion.msg.
		http://docs.ros.org/melodic/api/geometry_msgs/html/msg/Quaternion.html
		For the output, I will express in a row-principle manner.
		"""
		[x, y, z, w] = [quat.x, quat.y, quat.z, quat.w]
		rota = []
		# row 1
		rota.append(np.square(w) + np.square(x) - np.square(y) - np.square(z))
		rota.append(2*(x*y-w*z))
		rota.append(2*(x*z+w*y))
		# row 2
		rota.append(2*(x*y+w*z))
		rota.append(np.square(w) - np.square(x) + np.square(y) - np.square(z))
		rota.append(2*(y*z-w*x))
		# row 3
		rota.append(2*(x*z-w*y))
		rota.append(2*(y*z+w*x))
		rota.append(np.square(w) - np.square(x) - np.square(y) + np.square(z))
		return np.array(rota)

	def listener(self):
		"""
		This is our listener for 
		"""
		right_arm_state_sub = message_filters.Subscriber('/robot/limb/right/endpoint_state', EndpointState)
		rgb_state_sub = message_filters.Subscriber('/kinect2/sd/image_color_rect/compressed', CompressedImage)
		depth_state_sub = message_filters.Subscriber('/kinect2/sd/image_depth_rect/compressed', CompressedImage)
		ts = ApproxTimeSync([right_arm_state_sub, rgb_state_sub,  depth_state_sub], 1, 0.1)
		ts.registerCallback(self.listener_callback)

	def listener_callback(self, arm, rgb, depth):
		"""
		This method updates the variables.
		"""
		self.time = time.time()
		pose = arm.pose
		self.rgb = rgb.data
		self.depth = depth.data
		self.pos = pose.position
		self.orient = pose.orientation
		# Create input for net. x, y, z, rotation matrix
		queue_input = np.concatenate([np.array([self.pos.x, self.pos.y, self.pos.z]), self.quat_to_rotation(self.orient)])
		if len(self.queue) == 0:
			self.queue = [queue_input for i in range(5)]
		else:
			self.queue.pop(0)
			self.queue.append(queue_input)

	def get_limb_joints(self, output):
		"""
		This method gets the ik_solver solution for the arm joints.
		"""
		[goal_pos, goal_orient] = self.calculate_move(output[:3], output[3:])
		location = Point(*goal_pos)
		orientation = Quaternion(*goal_orient)
		limb_joints = ArmSolver('right').ik_solve(location, orientation)
		return limb_joints

	def calculate_move(self, lin, ang):
		"""
		This calculates the position and orientation (in quaterion) of the next pose given
		the linear and angular velocities outputted by the net.
		"""
		delta = self.time - self.prevTime
		# Position Update
		curr_pos = np.array([self.pos.x, self.pos.y, self.pos.z])
		goal_pos = np.add(curr_pos, delta*np.array(lin))
		# Orientation Update
		curr_orient = np.array([self.orient.x, self.orient.y, self.orient.z, self.orient.w])
		w_ang = np.concatenate([[0], ang])
		goal_orient = np.add(goal_orient, 0.5*delta*np.matmul(w_ang, np.transpose(curr_orient)))
		# Update the prevTime
		self.prevTime = self.time
		return goal_pos, goal_orient


upda
if __name__ == '__main__':
	learner = ImitateLearner()