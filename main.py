import os
import cv2
import numpy as np
import easyocr
import mysql.connector
import util
import matplotlib.pyplot as plt

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="parcare"
)


model_cfg = os.path.join('.', 'model', 'cfg', 'conf.cfg')
model_weights = os.path.join('.', 'model', 'weights', 'model.weights')
class_names = os.path.join('.', 'model', 'names', 'class.names')



cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

    H, W, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), True)

    net.setInput(blob)

    detections = util.get_outputs(net)

    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # SE APLICA NMS
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    reader = easyocr.Reader(['en'], gpu=True)
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        license_plate = frame[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

        frame = cv2.rectangle(frame,
                              (int(xc - (w / 2)), int(yc - (h / 2))),
                              (int(xc + (w / 2)), int(yc + (h / 2))),
                              (0, 255, 0),
                              15)

        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        output = reader.readtext(license_plate_thresh)
        cv2.imshow('License Plates Detection', frame)
        cv2.waitKey(0)
        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                print(text.replace(" ", ""), text_score)
                mycursor = mydb.cursor()

                mycursor.execute("SELECT * FROM usertable")

                myresult = mycursor.fetchall()

                ok_furat = 0

                for x in myresult:
                    if text.replace(" ", "") == x[7]:

                        print("Masina student ridica bariera")

                    else:

                        print("Masina nu student stai bariera")


    plt.figure()
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))

    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))

    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))

    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


