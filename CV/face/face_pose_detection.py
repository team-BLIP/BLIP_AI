'''
눈 10초이상 감는 것 도 인식 기능 만들어야 함 (완)
영상으로 온 것도 판단해서 구분해야 함
'''
import cv2
import mediapipe as mp

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 카메라 열기
cap = cv2.VideoCapture(0)

# Face Mesh 초기화
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("카메라를 열 수 없습니다.")
            break

        # 프레임을 거울처럼 반전 (좌우 반전)
        frame = cv2.flip(frame, 1)  # 1: 좌우 반전

        # BGR 이미지를 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Mediapipe Face Mesh 처리
        results = face_mesh.process(rgb_frame)

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

                # 눈 감지(제작 중)
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



                # 기준 크기 설정
                if face_size < 46:  # threshold_size는 초기 근접 거리 기준
                    cv2.putText(frame, "Too Far!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Close Enough!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)






                # 하단 기울기 감지
                if relative_tilt < 0.7 or relative_tilt > 3.43 or l_eyes < 3 or r_eyes < 3:  # 상대적으로 턱 끝과 가까워질수록 하단을 보고 있다고 판단
                    cv2.putText(frame, "BAD!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "GOOD!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



                # 기울기를 화면 왼쪽 하단에 표시
                text_tilt = f"Tilt: {relative_tilt:.2f}"  # 소수점 두 자리로 표시
                frame_height, frame_width, _ = frame.shape
                cv2.putText(frame, text_tilt, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



        # 결과 보여주기
        cv2.imshow('frame', frame)  # 창 이름 설정

        # Q 또는 q로 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Q 키를 눌러 프로그램을 종료합니다.")
            break

cap.release()
cv2.destroyAllWindows()