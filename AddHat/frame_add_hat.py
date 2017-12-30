# coding=utf-8
### Imports ###################################################################

from picamera.array import PiRGBArray
from picamera import PiCamera
from functools import partial

import multiprocessing as mp
import cv2
import os
import time


### Setup #####################################################################
# set env varible

os.putenv( 'SDL_FBDEV', '/dev/fb0' )

resX = 320
resY = 240
#resX = 640
#resY = 480


# Setup the camera
camera = PiCamera()
camera.resolution = ( resX, resY )
camera.framerate = 60

# Use this as our output
rawCapture = PiRGBArray( camera, size=( resX, resY ) )

# The face cascade file to be used
face_cascade = cv2.CascadeClassifier('../data/frontalface.xml')
eye_cascade = cv2.CascadeClassifier("../data/haarcascade_eye.xml")  
# 分离rgba通道，合成rgb三通道帽子图，a通道后面做mask用
r,g,b,a = cv2.split(cv2.imread("hat.png",-1)) 
rgb_hat = cv2.merge((r,g,b))

t_start = time.time()
fps = 0


### Helper Functions ##########################################################

def get_faces( img ):
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    faces = face_cascade.detectMultiScale( gray )
    return faces, img

def draw_frame( img, faces ):
    global fps
    global time_t
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	
    for (x,y,w,h) in faces:
        try: 
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)
            # eye
            face_gray = gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 2, cv2.CASCADE_SCALE_IMAGE, (30,30))
            if len(eyes) < 2:
            	return
            	
            #for(ex,ey,ew,eh) in eyes:
            #	cv2.rectangle(face_color, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)
            # 选取左右眼眼角的点
            (point1x,point1y,point1w,point1h) = eyes[0]
            (point2x,point2y,point2w,point2h) = eyes[1]
            # 求两点中心
            eyes_center = (x+(point1x+point1w+point2x)//2,y+(point1y+point2y)//2)
	
            cv2.circle(img,eyes_center,3,color=(255,0,0))  

            #  根据人脸大小调整帽子大小
            factor = 1.5
            resized_hat_h = int(round(rgb_hat.shape[0]*w/rgb_hat.shape[1]*factor))
            resized_hat_w = int(round(rgb_hat.shape[1]*w/rgb_hat.shape[1]*factor))

            if resized_hat_h > y:
            	resized_hat_h = y-1
	
            # 根据人脸大小调整帽子大小
            resized_hat = cv2.resize(rgb_hat,(resized_hat_w,resized_hat_h))

            # 用alpha通道作为mask
            mask = cv2.resize(a,(resized_hat_w,resized_hat_h))
		# inverse
            mask_inv =  cv2.bitwise_not(mask)

            # 帽子相对与人脸框上线的偏移量
            dh = +20
            dw = 0
            # 原图ROI
            # height=hat's height, wight=eyes_distance/3, and caclute the co
            bg_roi = img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)]
            # print bg_roi.shape[1]
            # print bg_roi.shape[0]
            # 原图ROI中提取放帽子的区域
            bg_roi = bg_roi.astype(float)
            mask_inv = cv2.merge((mask_inv,mask_inv,mask_inv))
            alpha = mask_inv.astype(float)/255
            # let roi region be masked
            # 相乘之前保证两者大小一致（可能会由于四舍五入原因不一致）
            alpha = cv2.resize(alpha,(bg_roi.shape[1],bg_roi.shape[0]))
            bg = cv2.multiply(alpha, bg_roi)
            bg = bg.astype('uint8')
	
            # 提取帽子区域
            hat = cv2.bitwise_and(resized_hat,resized_hat,mask = mask)
            # 相加之前保证两者大小一致（可能会由于四舍五入原因不一致）
            hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))
            # 两个ROI区域相加
            add_hat = cv2.add(bg,hat)

            # 把添加好帽子的区域放回原图
            img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)] = add_hat
        except BaseException:
	        return

    # Calculate and show the FPS
    fps = fps + 1
    sfps = fps / (time.time() - t_start)
    # cv2.putText(img, "FPS : " + str( int( sfps ) ), ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 ) 

    cv2.imshow( "Frame", img )
    cv2.waitKey( 1 )


### Main ######################################################################

if __name__ == '__main__':

	pool = mp.Pool( processes=4 )
	fcount = 0
	
	camera.capture( rawCapture, format="bgr" )  
	
	r1 = pool.apply_async( get_faces, [ rawCapture.array ] )    
	r2 = pool.apply_async( get_faces, [ rawCapture.array ] )    
	r3 = pool.apply_async( get_faces, [ rawCapture.array ] )    
	r4 = pool.apply_async( get_faces, [ rawCapture.array ] )    
	
	f1, i1 = r1.get()
	f2, i2 = r2.get()
	f3, i3 = r3.get()
	f4, i4 = r4.get()
	
	rawCapture.truncate( 0 )    
	
	for frame in camera.capture_continuous( rawCapture, format="bgr", use_video_port=True ):
		image = frame.array
		
		if   fcount == 1:
			r1 = pool.apply_async( get_faces, [ image ] )
			f2, i2 = r2.get()
			draw_frame( i2, f2 )
		
		elif fcount == 2:
			r2 = pool.apply_async( get_faces, [ image ] )
			f3, i3 = r3.get()
			draw_frame( i3, f3 )
		
		elif fcount == 3:
			r3 = pool.apply_async( get_faces, [ image ] )
			f4, i4 = r4.get()
			draw_frame( i4, f4 )
		
		elif fcount == 4:
			r4 = pool.apply_async( get_faces, [ image ] )
			f1, i1 = r1.get()
			draw_frame( i1, f1 )
		
			fcount = 0
	
		fcount += 1
	
		rawCapture.truncate( 0 )
