from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import base64
import json
import logging
import traceback
import os
import cv2
import mediapipe as mp
from uuid import uuid4
from PIL import Image
import io
import uvicorn

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 저장 디렉토리 설정
IMAGE_DIR = "./uploaded_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def face_detection():
    # Mediapipe 초기화
    mp_face_mesh = mp.solutions.face_mesh

    image_path = './uploaded_images/test.jpg'
    image = cv2.imread(image_path)

    if image is None or image.size == 0:
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


@app.post("/")
async def temp():
    return {"return": "hello"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info(f"WebSocket 연결 시도: {websocket.client}")
    await websocket.accept()
    logger.info(f"WebSocket 연결 성공: {websocket.client}")

    try:
        while True:
            try:
                # 클라이언트로부터 데이터 수신
                data = await websocket.receive_text()
                logger.info(f"받은 데이터: {data}")
                json_data = json.loads(data)

                user_id = json_data.get("id")
                base64_image = json_data.get("image")

                if not user_id or not base64_image:
                    await websocket.send_text(json.dumps({"error": "Invalid data format"}))
                    continue

                # 이미지 저장
                image_data = base64.b64decode(base64_image)
                img = Image.open(io.BytesIO(image_data))

                file_name = f"test.jpg"
                file_path = os.path.join(IMAGE_DIR, file_name)
                img.save(file_path)

                # 이미지 처리 시작
                logger.info(f"이미지 처리 시작: {file_path}")
                img_cv = cv2.imread(file_path)

                if img_cv is None:
                    logger.warning(f"이미지 로드 실패: {file_path}")
                    os.remove(file_path)  # 실패한 이미지 삭제
                    continue

                # OpenCV BGR → RGB 변환 후 저장
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                # 이미지 처리 함수에 이미지 전달
                res = face_detection()

                os.remove(file_path)  # 원본 이미지 삭제

                if res['situation'] == 1:
                    result = True
                else:
                    result = False

                # 처리 결과를 클라이언트에게 전송
                await websocket.send_text(json.dumps({"result": result}))

            except WebSocketDisconnect:
                logger.info(f"WebSocket 연결 종료: {websocket.client}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"JSON 디코딩 오류: {e}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
            except Exception as e:
                logger.error(f"데이터 처리 중 오류 발생: {e}")
                await websocket.send_text(json.dumps({"error": "Internal server error"}))

    except Exception as e:
        logger.error(f"WebSocket 연결 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"WebSocket 연결 종료: {websocket.client}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
