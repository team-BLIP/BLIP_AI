import cv2
import mediapipe as mp

def face_detection():
    # Mediapipe 초기화
    mp_face_mesh = mp.solutions.face_mesh

    image_path = './image/test.jpg'
    image = cv2.imread(image_path)

    if image is None or image.size == 0:u
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

    # 이미지 크기 저장
    height, width, _ = image.shape

    # Face Mesh 초기화
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        # BGR → RGB 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe Face Mesh 처리
        results = face_mesh.process(rgb_image)

        # 얼굴 랜드마크가 감지된 경우
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark = face_landmarks.landmark

                # 주요 랜드마크 (스크린 좌표 변환)
                nose_tip_y = landmark[4].y * height
                chin_tip_y = landmark[152].y * height
                forehead_tip_y = landmark[10].y * height

                # 얼굴 기울기 계산
                vertical_diff_down = abs(nose_tip_y - chin_tip_y)
                vertical_diff_up = abs(nose_tip_y - forehead_tip_y)
                relative_tilt = vertical_diff_down / vertical_diff_up

                # 눈 감김 감지 (픽셀 변환)
                l_eyes_h = landmark[159].y * height
                l_eyes_r = landmark[145].y * height
                r_eyes_h = landmark[386].y * height
                r_eyes_r = landmark[374].y * height

                l_eyes = abs(l_eyes_h - l_eyes_r)
                r_eyes = abs(r_eyes_h - r_eyes_r)

                # 얼굴 크기 계산
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]

                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                face_width = (max_x - min_x) * width
                face_height = (max_y - min_y) * height
                face_size = face_width * face_height

                result = {'far': 0, 'situation': 0}

                # 거리 판단 (픽셀 기준값 수정)
                if face_size < 5000:
                    result['far'] = 2  # 멀다
                elif face_size > 20000:
                    result['far'] = 1  # 가깝다

                # 기울기와 눈 감김 판단
                if (relative_tilt < 0.7 or relative_tilt > 3.43) and chin_tip_y > height * 0.85:
                    result['situation'] = 1  # 아래를 보고 있음

                if l_eyes < 5 or r_eyes < 5:
                    result['situation'] = 1  # 눈 감음

                # temp = {
                #     'far': 0,
                #     'situation': 1
                # }
                # print(temp)
                # return temp
                return result  # 결과 반환

temp2 = face_detection()
print(temp2)
