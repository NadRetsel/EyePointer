import mediapipe as mp
import numpy     as np
import pandas    as pd
import tkinter   as tk
import cv2
import time
import pyautogui


import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from threading import Thread

import recorder
import window

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

LEFT_EYE   = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS  = [474, 475, 476, 477]
LEFT_PUPIL = 473

RIGHT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_PUPIL = 468

HEAD_POSE = [33, 263, 1, 61, 291, 199]



AVERAGE_SIZE = 10


class Tracker:

    def __init__(self, mainWindow):
        self.mainWindow = mainWindow

        self.main_model = tf.keras.models.load_model('./model/main_model')
        self.left_closed_model = tf.keras.models.load_model('./model/left_closed_model')
        self.right_closed_model = tf.keras.models.load_model('./model/right_closed_model')

        self.main_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
        self.left_closed_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
        self.right_closed_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.rec = recorder.Recorder()

        self.rolling = [None] * AVERAGE_SIZE
        self.predictions = []
        self.ind = 0

        self.left_closed, self.right_closed, self.both_closed = False, False, False
        self.left_closed_timer, self.right_closed_timer, self.both_closed_timer = 0, 0, 0

        self.draw = False

    # Returns the face mesh landmarks
    def getFaceMesh(self):
        results = face_mesh.process(self.image)

        if not results.multi_face_landmarks:
            return None, None


        self.image_h, self.image_w, self.image_c = self.image.shape

        # Draw the mesh
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image                   = self.image,
                landmark_list           = face_landmarks,
                connections             = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec   = drawing_spec,
                connection_drawing_spec = drawing_spec
                )

            mesh_points = np.array([np.multiply([landmark.x, landmark.y, landmark.z], [self.image_w, self.image_h, 1]) for landmark in face_landmarks.landmark])

        return (results, mesh_points)


    # Returns the roll, pitch, yaw of head
    def getFaceAngles(self, mesh_points):
        image_h, image_w, image_c = self.image.shape
        face_3d = []
        face_2d = []

        # Get landmarks required for head pose estimation
        for idx, mesh_point in enumerate(mesh_points):
            if idx in HEAD_POSE:
                x,y,z = mesh_point
                face_2d.append([x,y])
                face_3d.append([x, y, z])

                if self.draw:
                    cv2.circle(self.image, (int(x), int(y)), 1, (0,0,255), 5)

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        cam_matrix = np.array([ [image_w, 0,       image_w / 2],
                                [0,       image_w, image_h / 2],
                                [0,       0,       1] ])

        distortion_matrix = np.zeros((4,1), dtype=np.float64)
        _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(cv2.Rodrigues(rot_vec)[0])

        x,y,z = np.array(angles) * 360

        if self.draw:
            if y > 10:
                text = "Looking Left"
            elif y < -10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Add the text on the image
            cv2.putText(self.image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.image, "x : " + str(np.round(x,2)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.image, "y : " + str(np.round(y,2)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.image, "z : " + str(np.round(z,2)), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return [x,y,z]


    # Returns minimum area rectangle of eyes + diameter of irises
    def getEyesLocation(self, mesh_points):
        image_h, image_w, image_c = self.image.shape

        # Convert location of landmarks to pixel coordiantes
        mesh_points_int = np.array(mesh_points[:, :2], dtype=np.int32)

        # Get minimum area rectangle around both eyes - [centre coordinates, size, angle of rotation]
        l_eye = cv2.minAreaRect(mesh_points_int[LEFT_EYE])
        r_eye = cv2.minAreaRect(mesh_points_int[RIGHT_EYE])

        # Calculate iris diameter
        l_iris_diam = (np.linalg.norm(mesh_points_int[LEFT_IRIS[0]]-mesh_points_int[LEFT_IRIS[1]]) + np.linalg.norm(mesh_points_int[LEFT_IRIS[2]]-mesh_points_int[LEFT_IRIS[3]])) / 2
        r_iris_diam = (np.linalg.norm(mesh_points_int[RIGHT_IRIS[0]]-mesh_points_int[RIGHT_IRIS[1]]) + np.linalg.norm(mesh_points_int[RIGHT_IRIS[2]]-mesh_points_int[RIGHT_IRIS[3]])) / 2


        if self.draw:
            l_box = np.int0(cv2.boxPoints(l_eye))
            cv2.drawContours(self.image, [l_box], 0, (0,255,0),2)
            #r_box = np.int0(cv2.boxPoints(r_eye))
            #cv2.drawContours(self.image, [r_box], 0, (0,255,0),2)


            cv2.line(self.image, mesh_points_int[LEFT_IRIS[0]], mesh_points_int[LEFT_IRIS[2]], (255, 255, 255), 2)
            cv2.line(self.image, mesh_points_int[LEFT_IRIS[1]], mesh_points_int[LEFT_IRIS[3]], (255, 255, 255), 2)
            #cv2.line(self.image, mesh_points[RIGHT_IRIS[0]], mesh_points[RIGHT_IRIS[2]], (255, 0, 0), 1)
            #cv2.line(self.image, mesh_points[RIGHT_IRIS[1]], mesh_points[RIGHT_IRIS[3]], (255, 0, 0), 1)


        return ([l_eye, r_eye], [l_iris_diam, r_iris_diam])


    # Returns distance of pupils from relative reference point
    def getPupilDists(self, mesh_points):

        l_hor, l_ver, _ = np.subtract(mesh_points[263], mesh_points[LEFT_PUPIL])
        r_hor, r_ver, _ = np.subtract(mesh_points[33], mesh_points[RIGHT_PUPIL])

        if self.draw:
            mesh_points_int = np.array(mesh_points[:, :2], dtype=np.int32)
            cv2.line(self.image, mesh_points_int[33], mesh_points_int[RIGHT_PUPIL], (255, 0, 0), 2)
            #cv2.line(self.image, mesh_points_int[33], mesh_points_int[LEFT_PUPIL], (0, 0, 255), 2)

        return ([l_hor, l_ver], [r_hor, r_ver])


    # Determine if a blink occurs
    def detectBlinks(self, eyes):
        l_eye, r_eye = eyes

        ratio_threshold = 7
        time_threshold = 0.75
        exit_threshold = 1.0

        l_width, l_height = max(l_eye[1]), min(l_eye[1])
        r_width, r_height = max(r_eye[1]), min(r_eye[1])
        l_ratio = l_width / l_height
        r_ratio = r_width / r_height


        # Detect eyes closing
        if(l_ratio >=  ratio_threshold and not self.left_closed):
            self.left_closed = True
            self.left_closed_timer = time.time()
            self.both_closed_timer = time.time()

        if(r_ratio >= ratio_threshold and not self.right_closed):
            self.right_closed = True
            self.right_closed_timer = time.time()
            self.both_closed_timer = time.time()

        if(self.left_closed and self.right_closed and not self.both_closed):
            self.both_closed = True
            self.both_closed_timer = time.time()


        if not self.both_closed:

            # Detect eyes opening for a single click
            if self.left_closed:
                if l_ratio <  ratio_threshold:
                    self.left_closed = False
                    if time.time() - self.left_closed_timer > time_threshold:
                        print("Left blink")
                        pyautogui.click(button = 'left')

                return "Left closed", "Single"

            if self.right_closed:
                if r_ratio <  ratio_threshold:
                    self.right_closed = False
                    if time.time() - self.right_closed_timer > time_threshold:
                        pyautogui.click(button = 'right')

                return "Right closed", "Single"

        else:
            # Exit mechanic
            if self.left_closed and self.right_closed and time.time() - self.both_closed_timer > exit_threshold:
                cv2.destroyAllWindows()
                self.left_closed = False
                self.right_closed = False
                self.both_closed = False
                self.mainWindow.toggleTracking()
                return "Exit", ""

            # Detect eyes opening (after both were closed) for click-and-hold
            result = "Both open", "Single"

            if l_ratio <  ratio_threshold:
                self.left_closed = False
                result =  "Right closed", "Double"

            if r_ratio <  ratio_threshold:
                self.right_closed = False
                result =  "Left closed", "Double"
            if (not self.left_closed) and (not self.right_closed):
                self.both_closed = False
                print("Blink")
                result = "Both open", "Single"

            return result


        return "Both open", "Single"



    # Returns the features for prediction
    def getDataPoints(self):
        _, self.image = self.cap.read()
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results,mesh_points = self.getFaceMesh()

        if results == None:
            return None, None


        # Get features
        angles = self.getFaceAngles(mesh_points)
        eyes, irises_diam = self.getEyesLocation(mesh_points)
        l_pupil_dist, r_pupil_dist = self.getPupilDists(mesh_points)
        model_type = self.detectBlinks(eyes)
        mouseCoord = pyautogui.position()

        return model_type, [r_pupil_dist[0], r_pupil_dist[1],   # Right pupil distances
                            l_pupil_dist[0], l_pupil_dist[1],   # Left pupil distances
                            1/irises_diam[1], 1/irises_diam[0], # Right, left iris distances from camera
                            angles[0], angles[1], angles[2],    # Face pose angles

                            mouseCoord[0], mouseCoord[1]]        # X and Y mouse coordinates



    def predict(self, isVisible):
        model_type, entry = self.getDataPoints()

        # Window that outputs camera feed
        if isVisible:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            self.draw = isVisible
            cv2.imshow('image', self.image)

        if entry == None or model_type[0] == "Exit":
            return

        entry = entry[:-2] # drop last 2 datapoints (mouse coordinates)

        # Take rolling average of 10 most recent frames
        self.rolling[self.ind] = entry
        self.ind += 1
        self.ind = self.ind % AVERAGE_SIZE
        if None in self.rolling:
            return
        self.average = np.mean(self.rolling, axis=0)

        # Use model depending on which eye(s) is available
        if self.left_closed:
            self.left_average = np.delete(self.average, [2,3,5])
            self.prediction = self.left_closed_model.predict(self.left_average, batch_size=50,verbose=0)[0]

        elif self.right_closed:
            self.right_average = np.delete(self.average, [0,1,4])
            self.prediction = self.right_closed_model.predict(self.right_average, batch_size=50,verbose=0)[0]

        else:
            self.prediction = self.main_model.predict(self.average, batch_size=50,verbose=0)[0]


        # Use mouse control based on how the user blinked
        if model_type[1] == "Single":
            pyautogui.moveTo(int(self.prediction[0]), int(self.prediction[1]))
        else:
            if model_type[0] == "Left closed":
                pyautogui.dragTo(int(self.prediction[0]), int(self.prediction[1]), 0.25, pyautogui.easeOutQuad, button='left')
            else:
                pyautogui.dragTo(int(self.prediction[0]), int(self.prediction[1]), 0.25, pyautogui.easeOutQuad, button='right')



    # Record 15 entries per calibration point
    def recordCalibration(self):
        for i in range(15):
            self.rec.addCalibrationEntry(self.getDataPoints()[1])

    # Train all 3 models (using threading)
    def beginCalibration(self):
        self.rec.calibrate()

        dataset = pd.read_csv('./model/calibrated.csv')

        X = dataset.drop(['Mouse x', 'Mouse y'], axis=1)
        X_left_closed = dataset.drop(['L hor pupil dist', 'L ver pupil dist', 'L iris cam dist', 'Mouse x', 'Mouse y'], axis=1)
        X_right_closed = dataset.drop(['R hor pupil dist', 'R ver pupil dist', 'R iris cam dist', 'Mouse x', 'Mouse y'], axis=1)

        Y = dataset.loc[:,['Mouse x', 'Mouse y']]

        # Split dataset to training and testing sets by 80:20
        X_train,       X_test,       Y_train,       Y_test = train_test_split(X, Y, test_size=0.2)
        X_left_train,  X_left_test,  Y_left_train,  Y_left_test = train_test_split(X_left_closed, Y, test_size=0.2)
        X_right_train, X_right_test, Y_right_train, Y_right_test = train_test_split(X_right_closed, Y, test_size=0.2)

        # Train the models in parallel
        def calibrateModels(self, model):
            if model == "Main":
                self.main_model.fit(X_train,               Y_train,       validation_split=0.2, verbose=1, epochs=100, batch_size=25)
            elif model == "Left closed":
                self.left_closed_model.fit(X_left_train,   Y_left_train,  validation_split=0.2, verbose=1, epochs=100, batch_size=25)
            else:
                self.right_closed_model.fit(X_right_train, Y_right_train, validation_split=0.2, verbose=1, epochs=100, batch_size=25)

        main  = Thread(target=calibrateModels, args=[self, "Main"])
        left  = Thread(target=calibrateModels, args=[self, "Left closed"])
        right = Thread(target=calibrateModels, args=[self, "Right closed"])

        main.start()
        left.start()
        right.start()

        main.join()
        left.join()
        right.join()
        print(self.main_model.evaluate(X_test, Y_test, verbose=0))
        print(self.left_closed_model.evaluate(X_left_test, Y_left_test, verbose=0))
        print(self.right_closed_model.evaluate(X_right_test, Y_right_test, verbose=0))

        calibratedFlag = False
