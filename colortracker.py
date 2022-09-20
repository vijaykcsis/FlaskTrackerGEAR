from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread

global capture,rec_frame, grey, switch, neg, face, rec, out, sl1, sl2, sl3, sl4, sl5, sl6
sl1=45
sl2=38
sl3=64
sl4=134
sl5=150
sl6=188
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    print("couldn't create screenshot dir")
    pass

#Load pretrained face detection model    
#net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

'''
def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
'''

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):
                #frame= detect_face(frame)
                pass
            if(grey):
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #tracker = cv2.TrackerCSRT_create()
                #bbox = cv2.selectROI(frame, True)

                print("SL1:", int(sl1)); print("SL2:", sl2); print("SL3:", sl3); print("SL4:", sl4); print("SL5:", sl5); print("SL6:", sl6);

                red_low = int(sl1)
                green_low = int(sl2)
                blue_low = int(sl3)

                red_hi = int(sl4)
                green_hi = int(sl5)
                blue_hi = int(sl6)

                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                hsv_lowerbound = np.array([red_low, green_low, blue_low])
                hsv_upperbound = np.array([red_hi, green_hi, blue_hi])

                print(hsv_lowerbound, hsv_upperbound)

                mask = cv2.inRange(hsv_frame, hsv_lowerbound, hsv_upperbound)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
                cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) > 0:
                  maxcontour = max(cnts, key=cv2.contourArea)
                  M = cv2.moments(maxcontour)
                  if M['m00'] > 0 and cv2.contourArea(maxcontour) > 1000:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.circle(frame, (cx, cy), 20, (255, 0, 0), -1)
                    #cv2.circle(frame, (cx, cy), 20, (0, 128, 255), -1)
                    cv2.circle(frame, (300, 700), 20, (0, 128, 255), -1)

                    cv2.line(frame, (cx, cy), (300, 700), (0, 0, 255), 1)
                    #cv2.line(frame, (color1_x, color1_y), (color2_x, color1_y), (0, 255, 0), 1)
                    #cv2.line(frame, (color2_x, color2_y), (color2_x, color1_y), (255, 0, 0), 1)


            if(neg):
                frame=cv2.bitwise_not(frame)
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html', slider1=sl1, slider2=sl2, slider3=sl3, slider4=sl4, slider5=sl5, slider6=sl6)
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests',methods=['POST','GET'])
def tasks():
    #text = request.form['sliderone']
    #processed_text = text.upper()
    #print(processed_text)

    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Activate Color-Based Tracking':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop Video':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
    elif request.method=='GET':
        return render_template('index.html', slider1=sl1, slider2=sl2, slider3=sl3, slider4=sl4, slider5=sl5, slider6=sl6)
    return render_template('index.html', slider1=sl1, slider2=sl2, slider3=sl3, slider4=sl4, slider5=sl5, slider6=sl6)

@app.route("/test", methods=["POST"])
def test():

      global sl1, sl2, sl3, sl4, sl5, sl6

      sl1 = request.form["slider1"]
      sl2 = request.form["slider2"]
      sl3 = request.form["slider3"]
      sl4 = request.form["slider4"]
      sl5 = request.form["slider5"]
      sl6 = request.form["slider6"]

      print("SL1:", sl1); print("SL2:", sl2); print("SL3:", sl3); print("SL4:", sl4); print("SL5:", sl5); print("SL6:", sl6);

      #return name_of_slider
      #if request.method=='GET':
          #return render_template('index.html')
      return render_template('index.html', slider1=sl1, slider2=sl2, slider3=sl3, slider4=sl4, slider5=sl5, slider6=sl6)


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()
