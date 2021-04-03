import cv2
import numpy as np 

def generate_OF(output_path, path_to_video):
    counter = 1
    cap = cv2.VideoCapture(path_to_video)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        if ret:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            flow_x = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
            flow_y = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
            flow_x = flow_x.astype('uint8')
            flow_y = flow_y.astype('uint8')

            flow_x = cv2.resize(src=flow_x, dsize=(224, 224))
            flow_y = cv2.resize(src=flow_y, dsize=(224, 224))

            cv2.imshow('horizontal'.format(counter), flow_x)
            cv2.imshow('vertical'.format(counter), flow_y)

            cv2.imwrite(output_path + "/" + 'flow_x_img{:05d}.jpg'.format(counter), flow_x)
            cv2.imwrite(output_path + "/" + 'flow_y_img{:05d}.jpg'.format(counter), flow_y)

            k = cv2.waitKey(30) & 0xff

            if k == 27:
                break
            
            prvs = next
            counter +=1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()