import cv2
import mediapipe as mp
import numpy     as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

LEFT_EYE   = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [469, 470, 471, 472]


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
        if y < -10:
            text = "Looking Left"
        elif y > 10:
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


def getEyesLocation(image, results):
    image_h, image_w, image_c = image.shape

    for face_landmarks in results.multi_face_landmarks:

        # Convert location of landmarks to pixel coordiantes
        mesh_points = np.array([np.multiply([landmark.x, landmark.y], [image_w, image_h]).astype(int) for idx,landmark in enumerate(face_landmarks.landmark)])

        # Get iris location
        (l_iris_x,l_iris_y),l_iris_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_iris_x,r_iris_y),r_iris_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        l_iris_centre = (int(l_iris_x),int(l_iris_y))
        r_iris_centre = (int(r_iris_x),int(r_iris_y))
        cv2.circle(image,l_iris_centre,int(l_iris_radius),(0,255,0),2)
        cv2.circle(image,r_iris_centre,int(r_iris_radius),(0,255,0),2)

        # Get eye box
        l_eye_x, l_eye_y, l_eye_w, l_eye_h = cv2.boundingRect(mesh_points[LEFT_EYE])
        r_eye_x, r_eye_y, r_eye_w, r_eye_h = cv2.boundingRect(mesh_points[RIGHT_EYE])
        cv2.rectangle(image,(l_eye_x, l_eye_y), (l_eye_x + l_eye_w, l_eye_y + l_eye_h), (0,255,0),2)
        cv2.rectangle(image,(r_eye_x, r_eye_y), (r_eye_x + r_eye_w, r_eye_y + r_eye_h), (0,255,0),2)

    return (image, results)


def main():
    cTime = 0
    pTime = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        pTime = cTime

        _, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
        image, results = getFaceMesh(image)

        if results.multi_face_landmarks:
            getFaceAngles(image, results)
            getEyesLocation(image, results)


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
