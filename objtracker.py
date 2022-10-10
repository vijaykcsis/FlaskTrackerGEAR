from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from math import dist
from math import acos
import math

global capture,rec_frame, grey, switch, neg, face, rec, out, sl1, sl2, sl3, sl4, sl5, sl6, tracker, firstTime, bboxCords, CONVFAC

sl1=45
tracker = cv2.TrackerCSRT_create()
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
firstTime=1
bboxCoords = [0, 0, 150, 150]
CONVFAC = 57.295779

def distance(x1, y1, x2, y2):
# Calculate distance between two points
  dist = math.sqrt(math.fabs(x2-x1)**2 + math.fabs(y2-y1)**2)
  return dist

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    print("couldn't create screenshot dir")
    pass

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame,  firstTime
    while True:
        success, frame = camera.read() 
        if success:
            if(face):
                #frame= detect_face(frame)
                pass
            if(grey):
                bbox = (bboxCoords[0], bboxCoords[1],         bboxCoords[2], bboxCoords[3])
                if (firstTime):
                  firstTime = 0
                  ok = tracker.init(frame, bbox)

                else:
                  #print(frame)
                  try:
                    ok, bbox = tracker.update(frame)
                  except Exception as e:
                    print("EXCEPTION", str(e))

                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

                centroid_x = int((p1[0]+p2[0])/2.0)
                centroid_y = int((p1[1]+p2[1])/2.0)

                print("CX, CY: ", centroid_x, centroid_y)

                anchor_x = 330
                anchor_y = 470


                A = distance(centroid_x, centroid_y, anchor_x, anchor_y)
                B = distance(anchor_x, anchor_y, anchor_x, centroid_y)
                C = distance(anchor_x, centroid_y, centroid_x, centroid_y)
                angle = -1.00
                if ((math.isnan(C)) or (math.isnan(B)) or (abs(B - 0.000) <= 0.01)):
                  angle = -1.00
                  acceleration = -1.00
                else:
                  angle = (np.arctan(C/B))
                  acceleration = (math.tan((2*angle)))*9.81

                cv2.putText(frame, ("Angle: "+ str(round(angle*57,3)) + " deg."), (abs(anchor_x),anchor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                cv2.putText(frame, ("Accel: "+ str(round(acceleration,3)) + " m/s^2"), (abs(anchor_x),anchor_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                print("ANGLE, ACCEL", angle*57, acceleration)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 255, 0), 5)
                cv2.circle(frame, (anchor_x, anchor_y), 5, (255, 255, 0), 5)
                cv2.line(frame, (centroid_x, centroid_y), (anchor_x, anchor_y), (255, 0, 0), 1)
                cv2.line(frame, (anchor_x, anchor_y), (anchor_x, centroid_y), (255, 0, 0), 1)
                cv2.line(frame, (anchor_x, centroid_y), (centroid_x, centroid_y), (255, 0, 0), 1)


                #bbox = cv2.selectROI(frame, True)
                '''
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
                '''
            else:
              bbox = (bboxCoords[0], bboxCoords[1],         bboxCoords[2], bboxCoords[3])
              p1 = (int(bbox[0]), int(bbox[1]))
              p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

              cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

              firstTime = 1
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
                #ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                ret, buffer = cv2.imencode('.jpg', frame)
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
        elif  request.form.get('grey') == 'Activate Tracker':
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

      #print("SL1:", sl1); print("SL2:", sl2); print("SL3:", sl3); print("SL4:", sl4); print("SL5:", sl5); print("SL6:", sl6);

      #return name_of_slider
      #if request.method=='GET':
          #return render_template('index.html')
      return render_template('index.html', slider1=sl1, slider2=sl2, slider3=sl3, slider4=sl4, slider5=sl5, slider6=sl6)


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()
