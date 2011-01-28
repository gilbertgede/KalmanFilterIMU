#!/usr/bin/python
import time
import os
from numpy import *
from kalmanFuncs import *


# This file has a simulated input into the Kalman filter in either the 7 or 13 state implementation
# Currently, the angular rates can only be constant


# Estimate the 9 sensor biases (13) or 3 biases (7)
numberStates = 7

# Environmental constants
g = 9.81
hloc = array([[0.225],[-0.057],[-0.4378]])

# Sensor noise constants
hnoise = array([0.01,0.01,0.01])
anoise = array([0.01,0.01,0.01])

# Process noise constants
hnoisep = array([0.001,0.001,0.001])
anoisep = array([0.001,0.001,0.001])
gnoisep = array([0.01,0.01,0.01])
wnoise = 0.02

# Simulation constants
t = 0.0
filterFreq = 50.0 #hz
dt = 1.0 / filterFreq
outputFreq = 1.0 #hz
counter = filterFreq / outputFreq
e1dot = 0.05
e2dot = 0.03
e3dot = 0.02

if numberStates == 7:
	hnoisep *= 0
	anoisep *= 0

# Simulation Noise & Bias
hnoises = array([0.01,0.01,0.01])
anoises = array([0.01,0.01,0.01])
gnoises = array([0.01,0.01,0.01])
hbiass = array([0.01,0.01,0.01])
abiass = array([0.01,0.01,0.01])
gbiass = array([0.05,0.05,0.05])

# Initial Conditions
X = array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
P = diag([0,0,0,0,0,0,0,0,0,0,0,0,0])
uu = array([[e1dot],[e2dot],[e3dot],[0],[0],[0],[0],[0],[0]])

# Main filter Loop
while (True):
	
	# Time update
	t += dt
	counter -= 1
	Q = Qmat(X,dt,wnoise,gnoisep,anoisep,hnoisep)
	A = Amat(X,uu,g,dt)
	X = Afunc(X,uu,g,dt)
	P = dot(dot(A,P),A.T) + Q
	H = Hmat(X,hloc,g)
	R = Rnmat(hnoise,anoise)
	K = dot(dot(P, H.T), linalg.inv(dot(dot(H, P), H.T) + R))
	YH = Hfunc(X,g,hloc)
	
	# Measurement generation
	e1 = arctan(tan(e1dot * t / 2.)) * 2
	e2 = arctan(tan(e2dot * t / 2.)) * 2
	e3 = arctan(tan(e3dot * t / 2.)) * 2
	q0 = cos(e1/2)*cos(e2/2)*cos(e3/2)+sin(e1/2)*sin(e2/2)*sin(e3/2)
	q1 = sin(e1/2)*cos(e2/2)*cos(e3/2)-cos(e1/2)*sin(e2/2)*sin(e3/2)
	q2 = cos(e1/2)*sin(e2/2)*cos(e3/2)+sin(e1/2)*cos(e2/2)*sin(e3/2)
	q3 = cos(e1/2)*cos(e2/2)*sin(e3/2)-sin(e1/2)*sin(e2/2)*cos(e3/2)
	w1 = e1dot * cos(e2) * cos(e3) + e2dot * sin(e3)
	w2 = - e1dot * cos(e2) * sin(e3) + e2dot * cos(e3)
	w3 = e1dot * sin(e2) + e3dot

# euler angle determination of rotation matrix - seems to require a negative sign on calculation of angles from estimated quaternion?
#	Tbi = array([[cos(e2)*cos(e3),-cos(e2)*sin(e3),sin(e2)],[sin(e1)*sin(e2)*cos(e3)+sin(e3)*cos(e3),-sin(e1)*sin(e2)*sin(e3)+cos(e3)*cos(e1),-sin(e1)*cos(e2)],[-cos(e1)*sin(e2)*cos(e3)+sin(e3)*sin(e1),cos(e1)*sin(e2)*sin(e3)+cos(e3)*sin(e1),cos(e1)*cos(e2)]])

# quaternion determination of rotation matrix
	Tbi = array([[1-2*(q2**2+q3**2),2*(q1*q2+q0*q3),2*(q1*q3-q0*q2)],[2*(q1*q2-q0*q3),1-2*(q1**2+q3**2),2*(q2*q3+q0*q1)],[2*(q1*q3+q0*q2),2*(q2*q3-q0*q1),1-2*(q1**2+q2**2)]])
	
	acc = array([Tbi[0][2]*g,Tbi[1][2]*g,Tbi[2][2]*g])
	acc[0] += random.normal(abiass[0],anoises[0])
	acc[1] += random.normal(abiass[1],anoises[1])
	acc[2] += random.normal(abiass[2],anoises[2])
	gyro = array([w1,w2,w3])
	gyro[0] += random.normal(gbiass[0],gnoises[0])
	gyro[1] += random.normal(gbiass[1],gnoises[1])
	gyro[2] += random.normal(gbiass[2],gnoises[2])
	mag = dot(Tbi,hloc)
	mag[0] += random.normal(hbiass[0],hnoises[0])
	mag[1] += random.normal(hbiass[1],hnoises[1])
	mag[2] += random.normal(hbiass[2],hnoises[2])
	Y = array([[mag[0,0]],[mag[1,0]],[mag[2,0]],[acc[0]],[acc[1]],[acc[2]]])
	uu = array([[gyro[0]],[gyro[1]],[gyro[2]],[acc[0]],[acc[1]],[acc[2]],[mag[0]],[mag[1]],[mag[2]]])
	
	# Measurement update
	X = X + dot(K,(Y-YH))
	P = dot((identity(13)-dot(K,H)),P)

	# Output generation
	if counter == 0:
                counter = filterFreq/outputFreq
		
		Eq0 = X[0][0]
		Eq1 = X[1][0]
		Eq2 = X[2][0]
		Eq3 = X[3][0]
		Ewpb= X[4][0]
		Ewqb= X[5][0]
		Ewrb= X[6][0]
		Eaxb= X[7][0]
		Eayb= X[8][0]
		Eazb= X[9][0]
		Ehxb= X[10][0]
		Ehyb= X[11][0]
		Ehzb= X[12][0]
		
		Ee1 = arctan2(2*(Eq0*Eq1+Eq2*Eq3),1-2*(Eq1**2+Eq2**2))
		Ee2 = arcsin(2*(Eq0*Eq2-Eq3*Eq1))
		Ee3 = arctan2(2*(Eq0*Eq3+Eq1*Eq2),1-2*(Eq2**2+Eq3**2))
		
		os.system("clear")
		print 'time - ',t
		print 'meas, calc, error'
		set_printoptions(precision = 4, linewidth = 120, suppress = True)
		print(Y.T)
		print(YH.T)
		print(Y.T-YH.T)
		
		print 'Euler angles'
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Ee1*180/pi,e1*180/pi,Ee1*180/pi-e1*180/pi)
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Ee2*180/pi,e2*180/pi,Ee2*180/pi-e2*180/pi)
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Ee3*180/pi,e3*180/pi,Ee3*180/pi-e3*180/pi)
		print sqrt(Eq0**2+Eq1**2+Eq2**2+Eq3**2)
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Eq0,q0,Eq0-q0)
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Eq1,q1,Eq1-q1)
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Eq2,q2,Eq2-q2)
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Eq3,q3,Eq3-q3) 
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Ewpb,gbiass[0],Ewpb-gbiass[0])
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Ewqb,gbiass[1],Ewqb-gbiass[1])
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Ewrb,gbiass[2],Ewrb-gbiass[2])
		if numberStates == 7:
			continue
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Eaxb,abiass[0],Eaxb-abiass[0])
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Eayb,abiass[1],Eayb-abiass[1])
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Eazb,abiass[2],Eazb-abiass[2])
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Ehxb,hbiass[0],Ehxb-hbiass[0])
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Ehyb,hbiass[1],Ehyb-hbiass[1])
		print 'est %.4f \t\t real %.4f \t\t error %.4f' % (Ehzb,hbiass[2],Ehzb-hbiass[2])


