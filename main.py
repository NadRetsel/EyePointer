import cv2
import mediapipe as mp
import numpy     as np
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

LEFT_EYE   = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_PUPIL = 473
#LEFT_IRIS  = [474, 475, 476, 477]

RIGHT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_PUPIL = 468
#RIGHT_IRIS = [469, 470, 471, 472]

l_closed = False
r_closed = False
b_closed = False


# Returns the landmarks
def getFaceMesh(image):
    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        return (image, results)

    image_h, image_w, image_c = image.shape

    # Draw the mesh
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image                   = image,
            landmark_list           = face_landmarks,
            connections             = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec   = drawing_spec,
            connection_drawing_spec = drawing_spec
            )

    return (image, results)


# Return the angles the face is tilted by
def getFaceAngles(image, results):
    image_h, image_w, image_c = image.shape
    face_3d = []
    face_2d = []

    # Calculate face pose angle
    for face_landmarks in results.multi_face_landmarks:
        for idx,landmark in enumerate(face_landmarks.landmark):
            if (idx == 33 or idx == 263) or (idx == 1) or (idx == 61 or idx == 291) or (idx == 199):
                if idx == 1:
                    # Used to make a line from nose tip
                    nose_2d = (landmark.x * image_w, landmark.y * image_h)
                    nose_3d = (landmark.x * image_w, landmark.y * image_h, landmark.z * 5000)

                # Landmark coordiantes converted from normalised form
                x, y = int(landmark.x * image_w), int(landmark.y * image_h)

                face_2d.append([x, y])
                face_3d.append([x, y, landmark.z])

        # Coordinates of important facepoints
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera matrix
        focal_length = image_w
        cam_matrix = np.array([ [focal_length, 0, image_h / 2],
                                [0, focal_length, image_w / 2],
                                [0, 0, 1] ])

        # 4x1 distortion matrix with all 0
        distortion_matrix = np.zeros((4,1), dtype=np.float64)

        # Solve PnP
        _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

        # Convert rot_vec vector into rotational matrix
        rot_mat, jac = cv2.Rodrigues(rot_vec)

        # Get angles from matrix
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_mat)

        # Convert normalised angles
        x,y,z = np.array(angles) * 360

        # See where the user's head tilting
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

        # Display the nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, distortion_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

        cv2.line(image, p1, p2, (255, 0, 0), 2)

    # Add the text on the image
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "x : " + str(np.round(x,2)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "y : " + str(np.round(y,2)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "z : " + str(np.round(z,2)), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return angles


# Returns coordinates of eyes and pupils
def getEyesLocation(image, results):
    image_h, image_w, image_c = image.shape

    for face_landmarks in results.multi_face_landmarks:

        # Convert location of landmarks to pixel coordiantes
        mesh_points = np.array([np.multiply([landmark.x, landmark.y], [image_w, image_h]).astype(int) for landmark in face_landmarks.landmark])

        # Draw box around eyes
        l_eye = cv2.minAreaRect(mesh_points[LEFT_EYE])
        l_box = np.int0(cv2.boxPoints(l_eye))
        cv2.drawContours(image, [l_box], 0, (0,255,0),2)

        r_eye = cv2.minAreaRect(mesh_points[RIGHT_EYE])
        r_box = np.int0(cv2.boxPoints(r_eye))
        cv2.drawContours(image, [r_box], 0, (0,255,0),2)

        # Display centre of boxes + pupil
        l_centre = (int(l_eye[0][0]), int(l_eye[0][1]))
        cv2.circle(image, l_centre, 1, (0,255,255), 2)
        cv2.circle(image, mesh_points[LEFT_PUPIL], 1, (255,0,0), 2)

        r_centre = (int(r_eye[0][0]), int(r_eye[0][1]))
        cv2.circle(image, r_centre, 1, (0,255,255), 2)
        cv2.circle(image, mesh_points[RIGHT_PUPIL], 1, (255,0,0), 2)

    return (l_eye, r_eye, [mesh_points[LEFT_PUPIL], mesh_points[RIGHT_PUPIL]])


# Returns the horizontal and vertical distances of pupils from centre of the eyes
def getPupilDists(l_eye, r_eye, pupils):

    # Calcualtes the horizontal and vertical distance of pupils from the centre, taking into account angle of tilt
    def calculatePupilDist(eye, pupil):
        manhat = np.subtract(eye[0], pupil)

        # Shift the range of the angles from [-90, 0), the range of minAreaRect, to [-45, 45)
        angle = (eye[2] + 45) % 90 - 45

        x_hor_component = manhat[0] * math.cos(angle)
        x_ver_component = manhat[0] * math.sin(angle)

        y_hor_component = manhat[1] * math.cos(angle)
        y_ver_component = manhat[1] * math.sin(angle)

        return( (x_hor_component + y_hor_component), (x_ver_component + y_ver_component))

    l_hor, l_ver = calculateEyeDist(l_eye, pupils[0])
    r_hor, r_ver = calculateEyeDist(r_eye, pupils[1])

    return ((l_hor, l_ver), (r_hor, r_ver))


# Calculate the width / height ratio of eyes to determine if a blink occurs
def detectBlinks(l_eye, r_eye):
    global l_closed
    global r_closed
    global b_closed

    ratio_threshold = 7

    l_width, l_height = max(l_eye[1]), min(l_eye[1])
    r_width, r_height = max(r_eye[1]), min(r_eye[1])
    l_ratio = l_width / l_height
    r_ratio = r_width / r_height


    if(not b_closed):
        # Detect eyes closing
        if(l_ratio >=  ratio_threshold and not l_closed):
            l_closed = True

        if(r_ratio >= ratio_threshold and not r_closed):
            r_closed = True

        if(l_closed and r_closed):
            b_closed = True

        # Detect eyes opening (only if both were not closed at the same time)
        if(l_ratio <  ratio_threshold and l_closed):
            l_closed = False
            print("Left blink")
        if(r_ratio <  ratio_threshold and r_closed):
            r_closed = False
            print("Right blink")

    else:
        # Detect when both eyes open to count as a blink
        if(l_ratio <  ratio_threshold and l_closed):
            l_closed = False
        if(r_ratio <  ratio_threshold and r_closed):
            r_closed = False
        if(not l_closed and not r_closed):
            b_closed = False



def main():
    cTime = 0
    pTime = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        pTime = cTime

        _, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, results = getFaceMesh(image)

        # If a face is detected
        if results.multi_face_landmarks:
            angles = getFaceAngles(image, results)
            l_eye, r_eye, pupils = getEyesLocation(image, results)
            l_dist, r_dist = getEyeDists(l_eye, r_eye, pupils)

            detectBlinks(l_eye, r_eye)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        cv2.putText(image, str(int(fps)), (10,400), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', image)

        k = cv2.waitKey(1)
        if k==27:
            break

    # Release the VideoCapture object
    cap.release()


if __name__ == "__main__":
    main()
