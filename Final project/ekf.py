import numpy as np 
from math import sqrt, sin, cos


class ekf():

	def __init__(self, x_init, beacons_pos, dt = 0.05):
		self.x = np.asarray(x_init).reshape(2,1) 
		self.x_predict = x_init
		self.P = np.array([[0.05**2,0],[0,0.05**2]])
		self.P_predict = self.P
		self.beacons = beacons_pos 
		self.dt = dt
		self.motion_noise = np.array([[0.03**2,0],[0,0.03**2]])
		self.sensor_noise = np.array([[0.01**2,0,0],[0,0.01**2,0],[0,0,0.01**2]])


	def compute_velocity_components(self, h):
		"""
		Given the youbot handle, compute the velocity components, derived from
		the inverse kinematics 
		"""
		wheels_radius = 0.0408

		A = np.array([[-1, -1, 1],[-1, 1, 1],[-1, -1, -1],[-1, 1,-1]])
		x = np.array([[h['previousForwBackVel']],[h['previousLeftRightVel']],[h['previousRotVel']]])

		#Compute the wheel velocities 
		omega = A@x
		A = np.array([[1,1,1,1],[-1,1,1,-1]])

		velocities = wheels_radius/4 * A@omega

		return velocities


	def predict(self, velocities, theta):
		"""
		Estimate the future state given the head orientation and the 
		velocity components
		velocities is a vector (vx vy)^T, theta is the robot orientation 
		"""

		#Compute the linear speed given vx, vy
		v = sqrt(velocities[0]**2 + velocities[1]**2)

		#Compute the elementary step 
		dd = v*self.dt 

		#Fx is the Jacobian of the motion model regarding the state
		Fx = np.array([[1,0],[0,1]])

		#Fv is the Jacobian of the motion model regarding the noise
		Fv = np.array([[cos(theta)],[sin(theta)]])

		#Compute the prediction
		self.x_predict = self.x + np.array([[dd*cos(theta)],[dd*sin(theta)]])

		#Compute predict the uncertainty
		self.P_predict = Fx@self.P@np.transpose(Fx) + self.motion_noise

		return 


	def update(self, observations):
		"""
		Given the observation from the sensors, update the estimate 
		observations is a 1x3 vector with the distances between youbot and the beacons 
		"""

		sensors_data = observations.reshape(3,1)

		#Compute the prediction on what the observations should be 
		projection = np.array([[sqrt((self.beacons[0][1]-self.x_predict[1][0])**2 + (self.beacons[0][0]-self.x_predict[0][0])**2)],\
							[sqrt((self.beacons[1][1]-self.x_predict[1][0])**2 + (self.beacons[1][0]-self.x_predict[0][0])**2)],\
							[sqrt((self.beacons[2][1]-self.x_predict[1][0])**2 + (self.beacons[2][0]-self.x_predict[0][0])**2)]])


		#Compute the error between the projection and observation, called nu 
		nu = sensors_data - projection

		#Hx is the Jacobian of the projection model, regarding the state
		Hx = np.array([[-(self.beacons[0][0]-self.x_predict[0][0])/projection[0][0], -(self.beacons[0][1]-self.x_predict[1][0])/projection[0][0]],\
					[-(self.beacons[1][0]-self.x_predict[0][0])/projection[1][0], -(self.beacons[1][1]-self.x_predict[1][0])/projection[1][0]],\
					[-(self.beacons[2][0]-self.x_predict[0][0])/projection[2][0], -(self.beacons[2][1]-self.x_predict[1][0])/projection[2][0]]]) 

		#Hw is the Jacobian of the projection model, regarding the noise 
		Hw = np.array([[1],[1],[1]])
		#print(Hw*self.sensor_noise*np.transpose(Hw))


		#Compute the parametric matrix S to later compute the kalman gain 
		S = np.array(Hx@self.P_predict@np.transpose(Hx) + self.sensor_noise)

		#Compute the kalman gain K 
		K = self.P_predict@np.transpose(Hx)@np.linalg.inv(S)

		#Finally update the state estimate and uncertainty estimate 
		self.x = self.x_predict + K@nu
		self.P = self.P_predict - K@Hx@self.P_predict

		return 
