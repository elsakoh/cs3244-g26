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

            horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
            vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
            horz = horz.astype('uint8')
            vert = vert.astype('uint8')

            # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # hsv[...,0] = ang*180/np.pi/2
            # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            cv2.imshow('horizontal'.format(counter), horz)
            cv2.imshow('vertical'.format(counter), vert)

            cv2.imwrite(output_path + "/" + 'flow_x_img{:05d}.jpg'.format(counter), horz)
            cv2.imwrite(output_path + "/" + 'flow_y_img{:05d}.jpg'.format(counter), vert)



            # cv2.imwrite(output_path + "/" + 'opticalfb_{:05d}.jpg'.format(counter),rgb)
            k = cv2.waitKey(30) & 0xff

            if k == 27:
                break
            # elif k == ord('s'):
                # cv2.imwrite(output_path + "/" + 'opticalhsv_{:05d}.jpg'.format(counter),rgb)
            prvs = next
            counter +=1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()