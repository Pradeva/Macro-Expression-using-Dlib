import cv2
import dlib
import math

# Inisialisasi detector wajah dan landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Fungsi untuk mendeteksi dan menggambar landmarks wajah
def detect_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    faceWithLandmarks = img.copy()  # Salin gambar agar tetap utuh
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(faceWithLandmarks, (x, y), 3, (0, 0, 255), -1)

        # SMILE DETECTION -------------------------
        
        # Kalkulasi Lebar Bibir
        lips_width = abs(landmarks.part(49).x - landmarks.part(55).x)

        # Kalkulasi Lebar Rahang
        jaw_width = abs(landmarks.part(3).x - landmarks.part(15).x)

        # Kalkulasi rasio bibir dan rahang
        lip_width_ratio = lips_width / jaw_width


        # SMILE DETECTION -------------------------

        # SURPRISED DETECTION -------------------------
        # Deteksi Mata Kiri
        left_eye_top = (landmarks.part(38).x, landmarks.part(38).y)
        left_eye_bot = (landmarks.part(40).x, landmarks.part(40).y)
        left_eye_outer =(landmarks.part(37).x, landmarks.part(37).y)
        left_eye_inner =(landmarks.part(38).x, landmarks.part(38).y)

        left_eye_height = calculate_distance(left_eye_top, left_eye_bot)
        left_eye_width = calculate_distance(left_eye_inner, left_eye_outer)

        eye_wide_ratio = left_eye_height / left_eye_width
            
        # Deteksi Mulut Terbuka

        outer_lip = abs(landmarks.part(51).y - landmarks.part(57).y)
        inner_lip = abs(landmarks.part(62).y - landmarks.part(66).y)

        if (outer_lip == 0):
            outer_lip = 0.00001

        lip_height_ratio = inner_lip / outer_lip

        # SURPRISED DETECTION -------------------------

        # ANGRY DETECTION -------------------------

        left_eyebrow_inner = (landmarks.part(21).x, landmarks.part(21).y)
        right_eyebrow_inner = (landmarks.part(22).x, landmarks.part(22).y)
        left_eye_inner = (landmarks.part(39).x, landmarks.part(39).y)
        right_eye_inner = (landmarks.part(42).x, landmarks.part(42).y)

        eyebrow_gap = calculate_distance(left_eyebrow_inner, right_eyebrow_inner)
        eye_gap = calculate_distance(left_eye_inner, right_eye_inner)

        eye_eyebrow_ratio = eyebrow_gap / eye_gap

        # ANGRY DETECTION -------------------------
            
        if (eye_wide_ratio > 0.89 and lip_height_ratio > 0.3):
            result = "Surprised"
        elif (lip_width_ratio > 0.3):
            result = "Smile"
        elif (eye_wide_ratio > 0.89 and eye_eyebrow_ratio < 0.4 and lip_height_ratio < 0.3):
            result = "Angry"
        else:
            result = ""
        
        # tambahkan teks dari nilai result
        cv2.putText(faceWithLandmarks, result, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return faceWithLandmarks

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Inisialisasi kamera
# cap = cv2.VideoCapture("video2.mp4")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Membalikkan frame secara horizontal
    frame_with_landmarks = detect_landmarks(frame.copy())
    
    cv2.imshow('Face Landmarks with Smile Detection', frame_with_landmarks)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
