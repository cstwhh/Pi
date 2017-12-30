# coding=utf-8
import numpy as np 
import cv2
# import dlib

# 给img中的人头像加上圣诞帽，人脸最好为正脸
def add_hat(img,hat_img):
    # 分离rgba通道，合成rgb三通道帽子图，a通道后面做mask用
    r,g,b,a = cv2.split(hat_img) 
    rgb_hat = cv2.merge((r,g,b))

    # # 灰度变换
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    # # 用opencv自带的人脸检测器检测人脸
    face_cascade = cv2.CascadeClassifier("../data/frontalface.xml")                       
    dets = face_cascade.detectMultiScale(gray,1.05,3,cv2.CASCADE_SCALE_IMAGE,(50,50))

    # 如果检测到人脸
    if len(dets)>0:  
        for (x,y,w,h) in dets:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)

            # eye
            eye_cascade = cv2.CascadeClassifier("../data/haarcascade_eye.xml")  
            face_gray = gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 2, cv2.CASCADE_SCALE_IMAGE, (30,30))
            if len(eyes) < 2:
            	return img
            	
            for(ex,ey,ew,eh) in eyes:
            	cv2.rectangle(face_color, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)
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

            return img

   
# 读取帽子图，第二个参数-1表示读取为rgba通道，否则为rgb通道
hat_img = cv2.imread("hat.png",-1)

# 读取头像图
img = cv2.imread("test.jpg")
output = add_hat(img,hat_img)

# 展示效果
cv2.imshow("output",output )  
cv2.waitKey(0)  
cv2.imwrite("output.jpg",output)

cv2.destroyAllWindows()  