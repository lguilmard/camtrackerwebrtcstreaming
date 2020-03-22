import numpy as np
# ~ import serial
import time
import sys
import cv2
import math as m 

import os


# ~ print(localPath)

def nothing(x):
	pass





def quatize(point,faces_pos):
	res = (int(min(faces_pos[0], key=lambda x:abs(x-point[0]))) , int(min(faces_pos[1], key=lambda x:abs(x-point[1]))))
	# ~ print(res)
	# ~ print(faces_pos[0],point[0])
	# ~ print(faces_pos[1],point[1])
	return res

def squares_CM(squares, ponderat=False, xshift=0, yshift=0):
	"""center of mass of squares"""
	suqares_centers = []
	suqares_weights = []
	CMX=[]
	CMY=[]
	for (x,y,w,h) in squares:
		CMX.append(x+w/2)
		CMY.append(y+h/2)
		suqares_weights.append([w*h])
	
	
	if ponderat == False:
		suqares_weights = [1]*len(squares)
	else:
		suqares_weights = [float(i)/sum(suqares_weights) for i in suqares_weights]
	
	# ~ print(suqares_weights,CMX,CMY)
	
	CMX=np.average(CMX,weights=suqares_weights)
	CMY=np.average(CMY,weights=suqares_weights)
	return (int(CMX+xshift),int(CMY+yshift))

def points_CM(points, ponderat=False, xshift=0, yshift=0):
	"""center of mass of squares"""
	suqares_centers = []
	suqares_weights = []
	CMX=[i[0] for i in points]
	CMY=[i[1] for i in points]
	
	if ponderat == False:
		suqares_weights = [1]*len(points)
	else:
		suqares_weights = [float(i)/sum(ponderat) for i in ponderat]
	
	# ~ print(suqares_weights,CMX,CMY)
	
	CMX=np.average(CMX,weights=suqares_weights)
	CMY=np.average(CMY,weights=suqares_weights)
	return (int(CMX+xshift),int(CMY+yshift))

def max_weight_point(points, ponderat):
	"""center of mass of squares"""
	pos = np.where(np.array(ponderat)== max(ponderat))[0][0]
	# ~ print(pos,points)
	selected = points[pos]
	return selected

def coord_index_list(point,weight=0,faces_pos=0):
	pos_X = np.where(np.array(faces_pos[0])== point[0])[0][0]
	list_positions_X = [0]*len(faces_pos[0])
	# ~ print(len(faces_pos[0]))
	list_positions_X[pos_X] = weight[0]
	# ~ print(pos_X,weight)
	pos_Y = np.where(np.array(faces_pos[1])== point[1])[0][0]
	list_positions_Y = [0]*len(faces_pos[1])
	# ~ print(len(faces_pos[1]))
	list_positions_Y[pos_Y] = weight[0]
	# ~ print(len(list_positions_X),len(list_positions_Y))
	return list_positions_X, list_positions_Y

def get_field(field,L,R,U,D,resolution):
	# is ROI inside field 
	ratio = resolution[1]/float(resolution[0])
	ratio_ROI = (L-R)/(D-U)
	if ratio_ROI < ratio: #portrait
		w = (D-U)*ratio
		dw = (w - (L-R))/2
		# ~ if R+w >= resolution[1]:
			# ~ field= (int(resolution[1]-w) , U , int(w) , (D-U))
		# ~ else:
		field= (int(min(max(0,R-dw),resolution[1]-w) ), U , int(w) , (D-U))
	elif ratio_ROI > ratio: #landscape
		h = (L-R)/ratio
		dh = (h - (D-U))
		field= (R, int(min(U+h , resolution[0])-h), (L-R),int( h))
	return field

def vote_dist_coor(vote,resolution):
	return



def cadreur(cap = cv2.VideoCapture(0)):
	print(sys.argv, len(sys.argv))
	localPath=str(os.path.dirname(os.path.realpath(__file__)))
	noArgs = len(sys.argv) == 1
	verbose = len(sys.argv) > 1 and sys.argv[1] == '-v'
	pipe_it = len(sys.argv) > 1 and sys.argv[1] == '-pipe'
	pipe_it_jpg = len(sys.argv) > 1 and sys.argv[1] == '-pipe_JPG'
	
	face_cascade = cv2.CascadeClassifier(localPath+'/XML_nn/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier(localPath+'/XML_nn/haarcascade_eye_tree_eyeglasses.xml')
	
	# ~ fake camera
	# ~ https://github.com/jremmons/pyfakewebcam
	
	cv2.namedWindow('cadreur')
	
	
	try:
		read_dictionary = np.load(localPath+'/setting.npy',allow_pickle='TRUE').item()
		NN_img_rescale = float(read_dictionary["NN_img_rescale"])
		Hxscale = float(read_dictionary["Hxscale"])
		sizeCare = float(read_dictionary["sizeCare"])
		decay = float(read_dictionary["decay"])
		print('setting file founded')
		setting = {"NN_img_rescale":NN_img_rescale, 
		"Hxscale":Hxscale,
		"sizeCare":sizeCare,
		"decay":decay}
		print(setting)
	except:
		NN_img_rescale = 1.35
		Hxscale = 2.4
		sizeCare = 0.5
		decay = 0.9
		setting = {"NN_img_rescale":NN_img_rescale, 
		"Hxscale":Hxscale,
		"sizeCare":sizeCare,
		"decay":decay}
		np.save(localPath+'/setting.npy', setting) 
		print('setting file not found -> default running')
	
	if verbose == True:
		cv2.namedWindow('image')
		# create trackbars for color change
		cv2.createTrackbar('NN_img_rescale(x100)','image',int(NN_img_rescale*100),450,nothing)
		cv2.createTrackbar('enlarge_height(x100)','image',int(Hxscale*100),450,nothing)
		cv2.createTrackbar('sizeCare(x100)','image',int(sizeCare*100),200,nothing)
		cv2.createTrackbar('decay(x100)','image',int(decay*100),100,nothing)
	
	ret, img = cap.read()
	# ~ img = cv2.imread('test2.png')
	resolution = img.shape
	maxface_H_W = (int(img.shape[0]/2),int(img.shape[1])/2)
	init = 10000
	
	faces_pos = [range(0,resolution[1],int(resolution[1]/maxface_H_W[1])), 
				 range(0,resolution[0],int(resolution[0]/maxface_H_W[0]))]
	
	# ~ print(faces_pos,resolution,maxface_H_W)
	
	vote_matrix = [
				np.array([0]*(len(faces_pos[0])-1)+[init]),
				np.array([init]+[0]*(len(faces_pos[0])-1)),
				np.array([init]+[0]*(len(faces_pos[1])-1)),
				np.array([0]*(len(faces_pos[1])-1)+[init]),
				(0,0,resolution[1],resolution[0])
				] # LimGauche LimDroite LimHaut LimBas ROI
	
	# ~ print(len(vote_matrix[0]),len(vote_matrix[2]))
	
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	TopLeftCornerOfText = (10,40)
	fontScale              = 1
	fontColor              = (0,0,255)
	lineType               = 2
	
	while 1:
		
		
		
		
		# ~ print(NN_img_rescale)
		
		ret, img = cap.read()
		# ~ img = cv2.imread('test2.png')
		img = cv2.flip( img, 1 )
		img_oigine = img.copy()
		
		if verbose == True:
			cv2.namedWindow('image', cv2.WINDOW_NORMAL)
			NN_img_rescale = cv2.getTrackbarPos('NN_img_rescale(x100)','image')/100
			Hxscale = cv2.getTrackbarPos('enlarge_height(x100)','image')/100
			sizeCare = cv2.getTrackbarPos('sizeCare(x100)','image')/100
			decay = cv2.getTrackbarPos('decay(x100)','image')/100
			setting = {"NN_img_rescale":NN_img_rescale, 
						"Hxscale":Hxscale,
						"sizeCare":sizeCare,
						"decay":decay}
		
		# mire
		# ~ cv2.line(img,(500,250),(0,250),(0,255,0),1)
		# ~ cv2.line(img,(250,0),(250,500),(0,255,0),1)
		# ~ cv2.circle(img, (250, 250), 5, (255, 255, 255), -1)
		
		gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, NN_img_rescale, 5)
		
		list_face_ROI_center = []
		list_face_ROI_left = []
		list_face_ROI_right = []
		list_face_ROI_up = []
		list_face_ROI_down = []
		list_face_ROI_weight = []
		
		
		
		#detect the face and make a rectangle around it.
		for (x,y,w,h) in faces:
			# ~ (x,y,w,h) = rounder_squares(square,resolution,maxface_W_H, devide_face=3)
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
			roi_gray  = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			
			# eyes
			# ~ eyes = eye_cascade.detectMultiScale(roi_gray, NN_img_rescale, 2)
			# ~ for (ex,ey,ew,eh) in eyes:
				# ~ cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,255),2)
			eyes = []
			
			if len(eyes) !=0:
				CM_eyes = squares_CM(eyes, xshift=x, yshift=y)
			else:
				CM_eyes = (int(x+w/2),int(y+h*1/3))
			
			linePoint1 = (x,CM_eyes[1])
			linePoint2 = (x+w,CM_eyes[1])
			if verbose == True:
				cv2.line(img,linePoint1,linePoint2,(0,255,0),1)
			
			# face ROI
			face_ROI_center = quatize((int(x+w/2),CM_eyes[1]),faces_pos)
			face_ROI_left = quatize((int(x+1.5*w),CM_eyes[1]),faces_pos)
			face_ROI_right = quatize((int(x-0.5*w),CM_eyes[1]),faces_pos)
			face_ROI_up = quatize((int(x+w/2),int(CM_eyes[1]-Hxscale*w*1/3)),faces_pos)
			face_ROI_down = quatize((int(x+w/2),int(CM_eyes[1]+Hxscale*w*2/3)),faces_pos)
			# ~ print(img.shape, quatize((int(x+w/2),int(CM_eyes[1]+Hxscale*w*2/3)),faces_pos), (int(x+w/2),int(CM_eyes[1]+Hxscale*w*2/3)) )
			face_ROI_weight = m.pow(h*w,sizeCare)
			
			#store 
			list_face_ROI_center.append(face_ROI_center)
			list_face_ROI_left.append(face_ROI_left)
			list_face_ROI_right.append(face_ROI_right)
			list_face_ROI_up.append(face_ROI_up)
			list_face_ROI_down.append(face_ROI_down)
			list_face_ROI_weight.append(face_ROI_weight)
			
			if verbose == True:
				cv2.circle(img, face_ROI_center, max(5,int(m.sqrt(h*w)*0.05)), (0, 0, 255), -1)
				
				# ~ # limtes droite gauche
				
				cv2.circle(img, face_ROI_left, max(5,int(m.sqrt(h*w)*0.05)), (255, 0, 0), -1)
				cv2.circle(img, face_ROI_right, max(5,int(m.sqrt(h*w)*0.05)), (255, 0, 0), -1)
				cv2.circle(img, face_ROI_up, max(5,int(m.sqrt(h*w)*0.05)), (255, 0, 0), -1)
				cv2.circle(img, face_ROI_down, max(5,int(m.sqrt(h*w)*0.05)), (255, 0, 0), -1)
		
		
		if len(faces)>0:
			
			faces_ROI = quatize(points_CM(list_face_ROI_center, list_face_ROI_weight), faces_pos)
			if verbose == True:
				cv2.circle(img, faces_ROI, max(5,int(m.sqrt(h*w)*0.05)), (255, 255, 0), -1)
			
			# si motorisation : correction
			#print( -(resolution[1]/2-faces_ROI[0]),resolution[0]/2-faces_ROI[1])
			if verbose == True:
				cv2.circle(img, (int(resolution[1]/2),int(resolution[0]/2)), 5, (255, 255, 255), -1)
			
			C = coord_index_list( faces_ROI ,list_face_ROI_weight, faces_pos)
			
			L = max_weight_point(list_face_ROI_left, np.array(list_face_ROI_weight)*[i[0] for i in list_face_ROI_left])
			# ~ cv2.line(img,(L[0],0),(L[0],img.shape[0]),(0,0,255),3)
			L = coord_index_list( L ,list_face_ROI_weight,faces_pos)[0]
			
			R = max_weight_point(list_face_ROI_right, np.array(list_face_ROI_weight)*[img.shape[1]-i[0] for i in list_face_ROI_right])
			# ~ cv2.line(img,(R[0],0),(R[0],img.shape[0]),(0,0,255),3)
			R = coord_index_list( R ,list_face_ROI_weight,faces_pos)[0]
			
			U = max_weight_point(list_face_ROI_up, np.array(list_face_ROI_weight)*[img.shape[0]-i[1] for i in list_face_ROI_up])
			# ~ cv2.line(img,(0,U[1]),(img.shape[1],U[1]),(0,0,255),3)
			U = coord_index_list( U ,list_face_ROI_weight,faces_pos)[1]
			
			D = max_weight_point(list_face_ROI_down, np.array(list_face_ROI_weight)*[i[1] for i in list_face_ROI_down])
			# ~ cv2.line(img,(0,D[1]),(img.shape[1],D[1]),(0,0,255),3)
			D = coord_index_list( D ,list_face_ROI_weight,faces_pos)[1]
			
			# ~ print(faces_pos)
			vote_matrix[0] = vote_matrix[0]*decay+np.array(L)
			vote_matrix[1] = vote_matrix[1]*decay+np.array(R)
			vote_matrix[2] = vote_matrix[2]*decay+np.array(U)
			vote_matrix[3] = vote_matrix[3]*decay+np.array(D)
			
			CL = int( np.average(faces_pos[0],weights=vote_matrix[0]) )
			CR = int( np.average(faces_pos[0],weights=vote_matrix[1]) )
			CU = int( np.average(faces_pos[1],weights=vote_matrix[2]) )
			CD = int( np.average(faces_pos[1],weights=vote_matrix[3]) )
			
			# ~ for i in range(len(vote_matrix)):
				# ~ print(len(faces_pos[0]),len(faces_pos[1]), i,vote_matrix[i].shape)
			vote_matrix[-1] = get_field(vote_matrix[-1],CL,CR,CU,CD,resolution) 
			
			Xfield = int( vote_matrix[-1][0]+vote_matrix[-1][2]/2 )
			Yfield = int( vote_matrix[-1][1]+vote_matrix[-1][3]/2 )
			
			## speed up recrop if out of the box
			# ~ print( -(Xfield-faces_ROI[0]),Yfield-faces_ROI[1])
			if abs(Xfield-faces_ROI[0]) > vote_matrix[-1][2]/2*2/3:
				# ~ print ("outofbox W")
				decay *= 0.8
			elif  abs(Yfield-faces_ROI[1]) > vote_matrix[-1][3]/2*2/3:
				# ~ print ("outofbox H")
				decay *= 0.8
			else:
				# ~ print ("inside the box <> decay = ",setting["decay"],decay)
				decay = setting["decay"]
			
			
			# ~ cv2.circle(img_oigine, (Xfield, Yfield), 5, (255, 255, 255), -1)
			# ~ cv2.circle(img_oigine, (faces_ROI[0], faces_ROI[1]), 5, (255, 255, 255), -1)
			
			# ~ print(C,L,R,U,D)  np.average(CMX,weights=suqares_weights)
			# ~ CL = int( np.average(faces_pos[0],weights=vote_matrix[0]) )
			# ~ CR = int( np.average(faces_pos[0],weights=vote_matrix[1]) )
			# ~ CU = int( np.average(faces_pos[1],weights=vote_matrix[2]) )
			# ~ CD = int( np.average(faces_pos[1],weights=vote_matrix[3]) )
			
			if verbose == True:
				cv2.line(img,(CL,0),(CL,img.shape[0]),(0,0,255),3)
				cv2.line(img,(CR,0),(CR,img.shape[0]),(0,0,255),3)
				cv2.line(img,(0,CU),(img.shape[1],CU),(0,0,255),3)
				cv2.line(img,(0,CD),(img.shape[1],CD),(0,0,255),3)
				
			# ~ print(vote_matrix[-1])
		
		# ~ print(resolution,vote_matrix)
		cv2.rectangle(img,(vote_matrix[-1][0],vote_matrix[-1][1]),(vote_matrix[-1][0]+vote_matrix[-1][2],vote_matrix[-1][1]+vote_matrix[-1][3]),(150,255,0),5)
		
		
		if verbose == True:
			ROI = cv2.resize(img_oigine[vote_matrix[-1][1]:vote_matrix[-1][1]+vote_matrix[-1][3], vote_matrix[-1][0]:vote_matrix[-1][0]+vote_matrix[-1][2]], (resolution[1],resolution[0])) 
			#Display the stream.
			# ~ cv2.imshow('image',cv2.resize(img, (0,0), fx=0.5, fy=0.5) )
			
			
			cv2.putText(img,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.putText(ROI,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.imshow('image-verbose',img )
			cv2.imshow('cadreur-verbose',ROI )
			#Hit 'Esc' to terminate execution
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		
		if noArgs:
			ROI = cv2.resize(img_oigine[vote_matrix[-1][1]:vote_matrix[-1][1]+vote_matrix[-1][3], vote_matrix[-1][0]:vote_matrix[-1][0]+vote_matrix[-1][2]], (resolution[1],resolution[0])) 
			#Display the stream.
			# ~ cv2.imshow('image',cv2.resize(img, (0,0), fx=0.5, fy=0.5) )
			cv2.putText(ROI,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.imshow('cadreur',ROI )
			
			#Hit 'Esc' to terminate execution
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		
		if pipe_it == True:
			ROI = cv2.resize(img_oigine[vote_matrix[-1][1]:vote_matrix[-1][1]+vote_matrix[-1][3], vote_matrix[-1][0]:vote_matrix[-1][0]+vote_matrix[-1][2]], (int(resolution[1]/4),int(resolution[0]/4))) 
			cv2.putText(ROI,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.imshow('cadreur-pipe-raw',ROI )
			
			#Hit 'Esc' to terminate execution
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
			
			sys.stdout.write(str(ROI.tostring()))
		
		if pipe_it_jpg == True:
			ROI = cv2.resize(img_oigine[vote_matrix[-1][1]:vote_matrix[-1][1]+vote_matrix[-1][3], vote_matrix[-1][0]:vote_matrix[-1][0]+vote_matrix[-1][2]], (int(resolution[1]/4),int(resolution[0]/4))) 
			cv2.putText(ROI,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.imshow('cadreur-pipe-jpg',ROI )
			ret, jpeg = cv2.imencode('.jpg', ROI)
			#Hit 'Esc' to terminate execution
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
			
			sys.stdout.write(str(jpeg.tobytes()))
	
	cap.release()
	cv2.destroyAllWindows()
	
	if verbose == True:
		print('setting file write' ,setting)
		
		np.save(localPath+'/setting.npy', setting) 


if __name__ == '__main__':
	cadreur(cv2.VideoCapture(0))
