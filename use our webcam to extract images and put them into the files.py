
import cv2
cap = cv2.VideoCapture(0)
i=0
while i<50:
    ret,frame=cap.read()
    cv2.imshow('capture',frame)
    cv2.imwrite('D:/AI_path/keras-frcnn-master/test_images/' + str(i) + ".jpg", frame)#将拍摄到的图片保存在test_images文件夹中
    i=i+1
    if cv2.waitKey(1000) == 27:  #&0xFF==ord('q'):#按键盘q就停止拍照,不按的话会收集满50张图自己停下 ，参数1000代表每1000ms（1s）收集一张图片
        break
cap.release()
cv2.destroyAllWindows()






# import cv2
#
# c=cv2.VideoCapture(0)
#
# while 1 :
#     ret, frame =c.read()
#     cv2.imshow('my', frame)
#     key=cv2.waitKey(1)
#     if key == 27:
#         cv2.destroyAllWindows()
#         break

