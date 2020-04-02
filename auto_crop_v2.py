import numpy as np
# ~ import serial
import time
import sys
import cv2
import math as m 

import os
import threading

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


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

vote_matrix_field = (0,0,10,10)
ContinueThreads = True

def cadreur(cap = cv2.VideoCapture(0)):
	global vote_matrix_field
	global ContinueThreads
	
	
	print(sys.argv, len(sys.argv))
	localPath=str(os.path.dirname(os.path.realpath(__file__)))
	noArgs = len(sys.argv) == 1
	threaded = len(sys.argv) > 1 and sys.argv[1] == '-thread'
	verbose = len(sys.argv) > 1 and sys.argv[1] == '-v'
	pipe_it = len(sys.argv) > 1 and sys.argv[1] == '-pipe'
	pipe_it_jpg = len(sys.argv) > 1 and sys.argv[1] == '-pipe_JPG'
	write =  len(sys.argv) > 1 and sys.argv[1] == "-write"
	
	face_cascade = cv2.CascadeClassifier(localPath+'/XML_nn/haarcascade_frontalface_alt.xml')
	eye_cascade = cv2.CascadeClassifier(localPath+'/XML_nn/haarcascade_eye_tree_eyeglasses.xml')
	
	# ~ fake camera
	# ~ https://github.com/jremmons/pyfakewebcam
	
	
	if threaded != True:
		cv2.namedWindow('cadreur')
	
	print('cadreur ON')
	try:
		read_dictionary = np.load(localPath+'/setting.npy',allow_pickle='TRUE').item()
		NN_img_rescale = float(read_dictionary["NN_img_rescale"])
		Hxscale = float(read_dictionary["Hxscale"])
		sizeCare = float(read_dictionary["sizeCare"])
		decay = float(read_dictionary["decay"])
		stabilizer = int(read_dictionary["stabilizer"])
		print('setting file founded')
		setting = {"NN_img_rescale":NN_img_rescale, 
		"Hxscale":Hxscale,
		"sizeCare":sizeCare,
		"decay":decay,
		"stabilizer":stabilizer}
		print(setting)
	except:
		NN_img_rescale = 1.30
		Hxscale = 2.7
		sizeCare = 0.27
		decay = 0.79
		stabilizer=15
		setting = {"NN_img_rescale":NN_img_rescale, 
		"Hxscale":Hxscale,
		"sizeCare":sizeCare,
		"decay":decay,
		"stabilizer":stabilizer}
		np.save(localPath+'/setting.npy', setting) 
		print('setting file not found -> default running')
	
	if verbose == True:
		cv2.namedWindow('image')
		
		# create trackbars for color change
		cv2.createTrackbar('stabilizer','image',int(stabilizer),20,nothing)
		cv2.createTrackbar('NN_img_rescale(x100)','image',int(NN_img_rescale*100),450,nothing)
		cv2.createTrackbar('enlarge_height(x100)','image',int(Hxscale*100),450,nothing)
		cv2.createTrackbar('sizeCare(x100)','image',int(sizeCare*100),200,nothing)
		cv2.createTrackbar('decay(x100)','image',int(decay*100),100,nothing)
	
	ret, img = cap.read()
	# ~ img = cv2.imread('test2.png')
	resolution = img.shape
	maxface_H_W = (int(img.shape[0]/1.1),int(img.shape[1])/1.1)
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
	
	vote_matrix_field = (0,0,resolution[1],resolution[0])
	
	threshold_newField_oldField = resolution[0]*0.05
	new_field = (0,0,resolution[1],resolution[0])
	
	# ~ print(len(vote_matrix[0]),len(vote_matrix[2]))
	
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	TopLeftCornerOfText = (10,40)
	fontScale              = 1
	fontColor              = (0,0,255)
	lineType               = 2
	
	outOfBox = False
	
	
	while ContinueThreads:
		# ~ try:
		
		# ~ print(NN_img_rescale)
		
		ret, img = cap.read()
		# ~ img = cv2.imread('test2.png')
		img = cv2.flip( img, 1 )
		img_oigine = img.copy()
		
		if verbose == True and outOfBox == False:
			cv2.namedWindow('image', cv2.WINDOW_NORMAL)
			NN_img_rescale = cv2.getTrackbarPos('NN_img_rescale(x100)','image')/100
			Hxscale = cv2.getTrackbarPos('enlarge_height(x100)','image')/100
			sizeCare = cv2.getTrackbarPos('sizeCare(x100)','image')/100
			decay = cv2.getTrackbarPos('decay(x100)','image')/100
			stabilizer = cv2.getTrackbarPos('stabilizer','image')
			setting = {"NN_img_rescale":NN_img_rescale, 
						"Hxscale":Hxscale,
						"sizeCare":sizeCare,
						"decay":decay,
						"stabilizer":stabilizer}
		
		# mire
		# ~ cv2.line(img,(500,250),(0,250),(0,255,0),1)
		# ~ cv2.line(img,(250,0),(250,500),(0,255,0),1)
		# ~ cv2.circle(img, (250, 250), 5, (255, 255, 255), -1)
		
		# ~ gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		# ~ faces = face_cascade.detectMultiScale(gray, NN_img_rescale, 5)
		
		# try https://github.com/kb22/Create-Face-Data-from-Images method  to get faces
		rescale = 2
		imgNN = cv2.resize(img, (int(resolution[1]/rescale),int(resolution[0]/rescale))) 
		(h, w) = imgNN.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(imgNN, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		
		model.setInput(blob)
		detections = model.forward()
		box = [detections[0, 0, i, 3:7] * np.array([w, h, w, h]).astype("int") for i in range(0, detections.shape[2])]
		confidence = [detections[0, 0, i, 2] for i in range(0, detections.shape[2])]
		faces = []
		for i in range(0, detections.shape[2]):
			if detections[0, 0, i, 2] > NN_img_rescale:
				# ~ print( detections[0, 0, i, 3:7] * np.array([w, h, w, h]).astype("int") ,i)
				face = rescale*detections[0, 0, i, 3:7] * np.array([w, h, w, h]).astype("int")
				faces.append( map(int,[face[0],face[1], face[2]-face[0], face[3]-face[1]]) )
		
		# ~ print(faces)
		# ~ faces = [ (box[i][0], box[i][1], box[i][0][3]-box[i][0][0], box[i][0][4]-box[i][0][2]) for i in range(0, detections.shape[2]) if confidence[i] > 0.5]
		# try https://github.com/kb22/Create-Face-Data-from-Images method  to get faces --> END
		
		list_face_ROI_center = []
		list_face_ROI_left = []
		list_face_ROI_right = []
		list_face_ROI_up = []
		list_face_ROI_down = []
		list_face_ROI_weight = []
		
		
		
		#detect the face and make a rectangle around it.
		for (x,y,w,h) in faces:
			# ~ (x,y,w,h) = rounder_squares(square,resolution,maxface_W_H, devide_face=3)
			# ~ print (x,y,w,h)
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
			# ~ roi_gray  = gray[y:y+h, x:x+w]
			# ~ roi_color = img[y:y+h, x:x+w]
			
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
			vote_matrix[0] = vote_matrix[0]*decay+np.array(L)*(1-decay)*min(2,len(faces))
			vote_matrix[1] = vote_matrix[1]*decay+np.array(R)*(1-decay)*min(2,len(faces))
			vote_matrix[2] = vote_matrix[2]*decay+np.array(U)*(1-decay)*min(2,len(faces))
			vote_matrix[3] = vote_matrix[3]*decay+np.array(D)*(1-decay)*min(2,len(faces))
			
			CL = int( np.average(faces_pos[0],weights=vote_matrix[0]) )
			CR = int( np.average(faces_pos[0],weights=vote_matrix[1]) )
			CU = int( np.average(faces_pos[1],weights=vote_matrix[2]) )
			CD = int( np.average(faces_pos[1],weights=vote_matrix[3]) )
			
			# ~ for i in range(len(vote_matrix)):
				# ~ print(len(faces_pos[0]),len(faces_pos[1]), i,vote_matrix[i].shape)
			new_field = get_field(vote_matrix[-1],CL,CR,CU,CD,resolution) 
			
			# ~ print(abs(CR-CL),abs(CU-CD),abs(CR-CL)*0.03)
			
			
				
			
			Xfield = int( new_field[0]+new_field[2]/2 )
			Yfield = int( new_field[1]+new_field[3]/2 )
			
			threshold_newField_oldField = abs(CR-CL)*max(1,stabilizer)/100.
			adjust_field = [abs(new_field[i]-vote_matrix[-1][i]) - threshold_newField_oldField*decay for i in range(len(new_field))]
			adjust_field = sum([i>0 for i in adjust_field]) > 0
			# ~ print(adjust_field_border, adjust_field)
			# ~ print(sum([abs(i) for i in adjust_field_border])/(threshold_newField_oldField*4),decay)
			
			## speed up recrop if out of the box
			# ~ print( -(Xfield-faces_ROI[0]),Yfield-faces_ROI[1])
			if adjust_field: #abs(Xfield-faces_ROI[0]) > new_field[2]/4 or abs(Yfield-faces_ROI[1]) > new_field[3]/4 
				# ~ print ("outofbox")
				outOfBox = True
				decay = max(0.1,0.6*decay)
			else:
				# ~ print ("inside the box <> decay = ",setting["decay"],decay)
				decay = setting["decay"]
				outOfBox = False
			
			# ~ print( abs(CR-CL)*0.03*decay ,decay)
			if adjust_field or stabilizer == 0:
				vote_matrix[-1] = new_field
				# ~ print('vote_matrix_field register')
				vote_matrix_field = new_field
			
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
		else:
			decay = setting["decay"]
		
		# ~ print(resolution,vote_matrix)
		cv2.rectangle(img,(new_field[0], new_field[1]), (new_field[0]+new_field[2], new_field[1]+new_field[3]),(0,255,255),int(threshold_newField_oldField*decay))
		
		cv2.rectangle(img,(vote_matrix[-1][0],vote_matrix[-1][1]),(vote_matrix[-1][0]+vote_matrix[-1][2],vote_matrix[-1][1]+vote_matrix[-1][3]),(150,255,0),5)
		
		if verbose == True:
			rescale = 2
			ROI = cv2.resize(img_oigine[vote_matrix[-1][1]:vote_matrix[-1][1]+vote_matrix[-1][3], vote_matrix[-1][0]:vote_matrix[-1][0]+vote_matrix[-1][2]], (int(resolution[1]/rescale),int(resolution[0]/rescale))) 
			cv2.putText(img,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.putText(ROI,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.imshow('image',img )
			cv2.imshow('cadreur',ROI )
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		
		if noArgs:
			rescale = 1
			cv2.rectangle(img_oigine,(new_field[0], new_field[1]), (new_field[0]+new_field[2], new_field[1]+new_field[3]),(0,255,255,0.1),int(threshold_newField_oldField*decay))
			cv2.rectangle(img_oigine,(0,0,resolution[1],resolution[0]),(0,0,255),15)
			ROI = cv2.resize(img_oigine[vote_matrix[-1][1]:vote_matrix[-1][1]+vote_matrix[-1][3], vote_matrix[-1][0]:vote_matrix[-1][0]+vote_matrix[-1][2]], (int(resolution[1]/rescale),int(resolution[0]/rescale))) 
			cv2.putText(ROI,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.imshow('cadreur',ROI )
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		
		if pipe_it == True:
			ROI = cv2.resize(img_oigine[vote_matrix[-1][1]:vote_matrix[-1][1]+vote_matrix[-1][3], vote_matrix[-1][0]:vote_matrix[-1][0]+vote_matrix[-1][2]], (int(resolution[1]/4),int(resolution[0]/4))) 
			cv2.putText(ROI,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.imshow('cadreur',ROI )
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
			
			sys.stdout.write(str(ROI.tostring()))
		
		if pipe_it_jpg == True:
			ROI = cv2.resize(img_oigine[vote_matrix[-1][1]:vote_matrix[-1][1]+vote_matrix[-1][3], vote_matrix[-1][0]:vote_matrix[-1][0]+vote_matrix[-1][2]], (int(resolution[1]/4),int(resolution[0]/4))) 
			cv2.putText(ROI,'PRESS ESC to exit', TopLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.imshow('cadreur',ROI )
			ret, jpeg = cv2.imencode('.jpg', ROI)
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
			
			sys.stdout.write(str(jpeg.tobytes()))
		
		if write == True:
			ROI = cv2.resize(img_oigine[vote_matrix[-1][1]:vote_matrix[-1][1]+vote_matrix[-1][3], vote_matrix[-1][0]:vote_matrix[-1][0]+vote_matrix[-1][2]], (int(resolution[1]/4),int(resolution[0]/4))) 
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
			encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
			cv2.imwrite('test.png', ROI, encode_param)
			
		# ~ except:
			# ~ print('error cadrer')
			# ~ pass
	
	
	cv2.destroyAllWindows()
	
	if verbose == True:
		print('setting file write' ,setting)
		
		np.save(localPath+'/setting.npy', setting) 

def write_cap(cap = cv2.VideoCapture(0)):
	global vote_matrix_field
	global ContinueThreads
	localPath=str(os.path.dirname(os.path.realpath(__file__)))
	iteration = 0
	while ContinueThreads:
		iteration+=1
		
		ret, img = cap.read()
		# ~ img = cv2.imread('test2.png')
		
		try:
			img = cv2.flip( img, 1 )
			ROI = cv2.resize(img[vote_matrix_field[1]:vote_matrix_field[1]+vote_matrix_field[3], vote_matrix_field[0]:vote_matrix_field[0]+vote_matrix_field[2]], (int(resolution[1]),int(resolution[0]))) 
			encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
			cv2.imwrite(localPath+'/tmpfs_DIR/auto_crop_output.png', ROI, encode_param)
		except:
			print("error")
			
	cap.release()

def thread_killer():
	global ContinueThreads
	input("Press Enter to STOP...")
	ContinueThreads = False
	print('STOP CADREUR')

if __name__ == '__main__':
	thread = len(sys.argv) > 1 and sys.argv[1] == '-thread'
	ContinueThreads = True
	
	if thread:
		cap = cv2.VideoCapture(0)
		ret, img = cap.read()
		resolution = img.shape
		cap.release()
		# ~ print(ContinueThreads)
		
		fieldUpdater = threading.Thread(target=cadreur)
		fieldUpdater.start()
		croper = threading.Thread(target=write_cap)
		croper.start()
		print(ContinueThreads)
		threadkiller = threading.Thread(target= thread_killer)
		threadkiller.start()
		
		# ~ print(vote_matrix_field)
	else:
		cadreur()
	
