import cv2
import mediapipe as mp

def main():


    # Mediapipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    # 이미지 파일 경로 설정
    image_path = r"C:\folders\project\BLIP_AI_main\CV\face\image\WIN_20250307_15_23_34_Pro.jpg"  # 여기에 분석할 이미지 파일 경로를

    image = cv2.imread(image_path)

    if image is None or image.size == 0:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

    # Face Mesh 초기화
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        # 이미지의 가로, 세로 크기 저장
        height, width, _ = image.shape

        # BGR 이미지를 RGB로 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe Face Mesh 처리
        results = face_mesh.process(rgb_image)

        # 얼굴 랜드마크가 감지된 경우
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 랜드마크 가져오기
                landmark = face_landmarks.landmark

                # 랜드마크 인덱스
                nose_tip = landmark[4]  # 코 끝
                chin_tip = landmark[152]  # 턱 끝
                forehead_tip = landmark[10]  # 이마(위쪽 중앙)

                # 기울기 계산 (코 끝과 턱 끝 vs 코 끝과 이마 중앙)
                vertical_diff_down = abs(nose_tip.y - chin_tip.y)  # 코 끝과 턱 끝의 세로 거리
                vertical_diff_up = abs(nose_tip.y - forehead_tip.y)  # 코 끝과 이마 중앙의 세로 거리

                # 하단과 상단 세로 거리 비교를 통해 기울기 계산
                relative_tilt = vertical_diff_down / vertical_diff_up

                # 눈 감지
                l_eyes_h = landmark[159]
                l_eyes_r = landmark[145]

                r_eyes_h = landmark[386]
                r_eyes_r = landmark[374]

                l_eyes_h_y = l_eyes_h.y * 1000
                l_eyes_r_y = l_eyes_r.y * 1000

                r_eyes_h_y = r_eyes_h.y * 1000
                r_eyes_r_y = r_eyes_r.y * 1000

                l_eyes = abs(l_eyes_h_y - l_eyes_r_y)
                r_eyes = abs(r_eyes_h_y - r_eyes_r_y)

                # 랜드마크 x, y 좌표 가져오기
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]

                # 최소, 최대 좌표로 bounding box 계산
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                # bounding box 크기 계산
                face_width = max_x - min_x
                face_height = max_y - min_y

                face_size = face_width * face_height * 1000

                chack = {
                    'far':0,
                    'situation':0,
                }

                # 기준 크기 설정
                if face_size < 46:
                    chack['far'] = 2 #먼

                elif face_size > 350:
                    chack['far'] = 1 #가까운


                # 하단 기울기 감지
                if relative_tilt < 0.7 or relative_tilt > 3.43 and chin_tip > 18:  # 상대적으로 턱 끝과 가까워질수록 하단을 보고 있다고 판단
                    chack['situation'] = 1


                if l_eyes < 11 or r_eyes < 11:
                    chack['situation'] = 1

                return chack

print(main())