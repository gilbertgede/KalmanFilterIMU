#!/usr/bin/python
import numpy as np
import time


# This is the linearized matrix of the state space function
def Amat(x,uu,g,dt):
	q0=x[0][0]
	q1=x[1][0]
	q2=x[2][0]
	q3=x[3][0]
	wpb=x[4][0]
	wqb=x[5][0]
	wrb=x[6][0]
	axb=x[7][0]
	ayb=x[8][0]
	azb=x[9][0]
	hxb=x[10][0]
	hyb=x[11][0]
	hzb=x[12][0]

	wp=uu[0][0]
	wq=uu[1][0]
	wr=uu[2][0]
	ax=uu[3][0]
	ay=uu[4][0]
	az=uu[5][0]
	hx=uu[6][0]
	hy=uu[7][0]
	hz=uu[8][0]
	
	A = np.zeros((13,13))
	A = np.identity(13) * -2 / dt
	A[0][1]=wp-wpb
	A[0][2]=wq-wqb
	A[0][3]=wr-wrb
	A[0][4]=-q1
	A[0][5]=-q2
	A[0][6]=-q3
	A[1][0]=-wp+wpb
	A[1][2]=-wr+wrb
	A[1][3]=wq-wqb
	A[1][4]=q0
	A[1][5]=-q3
	A[1][6]=q2
	A[2][0]=-wq+wqb
	A[2][1]=wr-wrb
	A[2][3]=-wp+wpb
	A[2][4]=q3
	A[2][5]=q0
	A[2][6]=-q1
	A[3][0]=-wr+wrb
	A[3][1]=-wq+wqb
	A[3][2]=wp-wpb
	A[3][4]=-q2
	A[3][5]=q1
	A[3][6]=q0
	A = A * dt * -0.5
	return A


# This is the linearized matrix of the measurement function
def Hmat(x,hloc,g):
	q0=x[0][0]
	q1=x[1][0]
	q2=x[2][0]
	q3=x[3][0]
	wpb=x[4][0]
	wqb=x[5][0]
	wrb=x[6][0]	
	axb=x[7][0]
	ayb=x[8][0]
	azb=x[9][0]
	hxb=x[10][0]
	hyb=x[11][0]
	hzb=x[12][0]

	H=np.zeros((6,13))
	H[0][0] = -2*hloc[2]*q2+2*hloc[1]*q3
	H[0][1] = 2*hloc[1]*q2+2*hloc[2]*q3
	H[0][2] = -2*hloc[2]*q0+2*hloc[1]*q1-4*hloc[0]*q2
	H[0][3] = 2*hloc[1]*q0+2*hloc[2]*q1-4*hloc[0]*q3
	H[0][10] = 1
	H[1][0] = 2*hloc[2]*q1-2*hloc[0]*q3
	H[1][1] = 2*hloc[2]*q0-4*hloc[1]*q1+2*hloc[0]*q2
	H[1][2] = 2*hloc[0]*q1+2*hloc[2]*q3
	H[1][3] = -2*hloc[0]*q0+2*hloc[2]*q2-4*hloc[1]*q3
	H[1][11] = 1
	H[2][0] = -2*hloc[1]*q1+2*hloc[0]*q2
	H[2][1] = -2*hloc[1]*q0-4*hloc[2]*q1+2*hloc[0]*q3
	H[2][2] = 2*hloc[0]*q0-4*hloc[2]*q2+2*hloc[1]*q3
	H[2][3] = 2*hloc[0]*q1+2*hloc[1]*q2
	H[2][12] = 1
	H[3][0] = -2*g*q2
	H[3][1] = 2*g*q3
	H[3][2] = -2*g*q0
	H[3][3] = 2*g*q1
	H[3][7] = 1
	H[4][0] = 2*g*q1
	H[4][1] = 2*g*q0
	H[4][2] = 2*g*q3
	H[4][3] = 2*g*q2
	H[4][8] = 1
	H[5][1] = -4*g*q1
	H[5][2] = -4*g*q2
	H[5][9] = 1
	return H


# The actual state space update function
def Afunc(x,uu,g,dt):
	q0=x[0][0]
	q1=x[1][0]
	q2=x[2][0]
	q3=x[3][0]
	wpb=x[4][0]
	wqb=x[5][0]
	wrb=x[6][0]
	axb=x[7][0]
	ayb=x[8][0]
	azb=x[9][0]
	hxb=x[10][0]
	hyb=x[11][0]
	hzb=x[12][0]
	
	wp=uu[0][0]
	wq=uu[1][0]
	wr=uu[2][0]
	ax=uu[3][0]
	ay=uu[4][0]
	az=uu[5][0]
	hx=uu[6][0]
	hy=uu[7][0]
	hz=uu[8][0]
	
	gyrp = wp - wpb
	gyrq = wq - wqb
	gyrr = wr - wrb

	Tbi = np.array([[1-2*(q2**2+q3**2),2*(q1*q2+q0*q3),2*(q1*q3-q0*q2)],[2*(q1*q2-q0*q3),1-2*(q1**2+q3**2),2*(q2*q3+q0*q1)],[2*(q1*q3+q0*q2),2*(q2*q3-q0*q1),1-2*(q1**2+q2**2)]])
	
	A = np.zeros((13,1))
	phi = np.array([[0,gyrp,gyrq,gyrr],[-gyrp,0,-gyrr,gyrq],[-gyrq,gyrr,0,-gyrp],[-gyrr,-gyrq,gyrp,0]])*dt
	s = .5 * np.sqrt(gyrp**2+gyrq**2+gyrr**2)
	jj = 1 - np.sqrt(q0**2+q1**2+q2**2+q3**2)
	AAA = np.dot((np.identity(4)*(np.cos(s)+jj*.99999)-.5*phi*(np.sin(s)/s)),np.array([[q0],[q1],[q2],[q3]]))
	A[0] = AAA[0][0]
	A[1] = AAA[1][0]
	A[2] = AAA[2][0]
	A[3] = AAA[3][0]
	A[4] = wpb
	A[5] = wqb
	A[6] = wrb
	A[7] = axb
	A[8] = ayb
	A[9] = azb
	A[10] = hxb
	A[11] = hyb
	A[12] = hzb
	return A


# The measurement function
def Hfunc(x,g,hloc):
	q0=x[0][0]
	q1=x[1][0]
	q2=x[2][0]
	q3=x[3][0]
	wpb=x[4][0]
	wqb=x[5][0]
	wrb=x[6][0]
	axb=x[7][0]
	ayb=x[8][0]
	azb=x[9][0]
	hxb=x[10][0]
	hyb=x[11][0]
	hzb=x[12][0]

	Tbi = np.array([[1-2*(q2**2+q3**2),2*(q1*q2+q0*q3),2*(q1*q3-q0*q2)],[2*(q1*q2-q0*q3),1-2*(q1**2+q3**2),2*(q2*q3+q0*q1)],[2*(q1*q3+q0*q2),2*(q2*q3-q0*q1),1-2*(q1**2+q2**2)]])

	H = np.zeros((6,1))
	H[0] = (np.dot(Tbi[0],hloc)+hxb)	
	H[1] = (np.dot(Tbi[1],hloc)+hyb)
	H[2] = (np.dot(Tbi[2],hloc)+hzb)
	H[3] = Tbi[0][2]*g+axb
	H[4] = Tbi[1][2]*g+ayb
	H[5] = Tbi[2][2]*g+azb
	return H


# The process noise covariance matrix
def Qmat(x,dt,wnoise,gnoise,anoise,hnoise):
	q0=x[0][0]
	q1=x[1][0]
	q2=x[2][0]
	q3=x[3][0]
	wpb=x[4][0]
	wqb=x[5][0]
	wrb=x[6][0]	
	axb=x[7][0]
	ayb=x[8][0]
	azb=x[9][0]
	hxb=x[10][0]
	hyb=x[11][0]
	hzb=x[12][0]
	
	hx = hnoise[0]
	hy = hnoise[1]
	hz = hnoise[2]
	ax = anoise[0]
	ay = anoise[1]
	az = anoise[2]
	gx = gnoise[0]
	gy = gnoise[1]
	gz = gnoise[2]

	Q = np.zeros((13,13))
	
	Q[0][0] = ((.5*dt*wnoise)**2)*(q1**2+q2**2+q3**2)
	Q[0][1] = ((.5*dt*wnoise)**2)*(-q0*q1)
	Q[0][2] = ((.5*dt*wnoise)**2)*(-q0*q2)
	Q[0][3] = ((.5*dt*wnoise)**2)*(-q0*q3)
	Q[1][0] = ((.5*dt*wnoise)**2)*(-q0*q1)
	Q[1][1] = ((.5*dt*wnoise)**2)*(q0**2+q2**2+q3**2)
	Q[1][2] = ((.5*dt*wnoise)**2)*(-q1*q2)
	Q[1][3] = ((.5*dt*wnoise)**2)*(-q1*q3)
	Q[2][0] = ((.5*dt*wnoise)**2)*(-q0*q2)
	Q[2][1] = ((.5*dt*wnoise)**2)*(-q1*q2)
	Q[2][2] = ((.5*dt*wnoise)**2)*(q0**2+q1**2+q3**2)
	Q[2][3] = ((.5*dt*wnoise)**2)*(-q2*q3)
	Q[3][0] = ((.5*dt*wnoise)**2)*(-q0*q3)
	Q[3][1] = ((.5*dt*wnoise)**2)*(-q1*q3)
	Q[3][2] = ((.5*dt*wnoise)**2)*(-q2*q3)
	Q[3][3] = ((.5*dt*wnoise)**2)*(q0**2+q1**2+q2**2)
	Q[4][4] = dt*gx**2
	Q[5][5] = dt*gy**2
	Q[6][6] = dt*gz**2
	Q[7][7] = dt*ax**2
	Q[8][8] = dt*ay**2
	Q[9][9] = dt*az**2
	Q[10][10] = dt*hx**2
	Q[11][11] = dt*hy**2
	Q[12][12] = dt*hz**2
	return Q


# The sensor noise covariance matrix
def Rnmat(hnoise,anoise):
	Rn = np.zeros((6,6))
	hx = hnoise[0]
	hy = hnoise[1]
	hz = hnoise[2]
	ax = anoise[0]
	ay = anoise[1]
	az = anoise[2]
	Rn = np.zeros((6,6))
	Rn[0][0] = hx**2
	Rn[1][1] = hy**2
	Rn[2][2] = hz**2
	Rn[3][3] = ax**2
	Rn[4][4] = ay**2
	Rn[5][5] = az**2
	return Rn


