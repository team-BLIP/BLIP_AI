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

                json_data = json.loads(data)

                user_id = json_data.get("id")
                base64_image = json_data.get("image")

                if not user_id or not base64_image:
                    await websocket.send_text(json.dumps({"result": False}))
                    continue

                # 이미지 저장
                image_data = base64.b64decode(base64_image)
                img = Image.open(io.BytesIO(image_data))

                file_name = f"{uuid4()}.jpg"
                file_path = os.path.join(IMAGE_DIR, file_name)
                img.save(file_path)

                # 이미지 처리 시작
                logger.info(f"이미지 처리 시작: {file_path}")
                res = face_detection(file_path)  # 동적으로 이미지 경로 전달

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
                await websocket.send_text(json.dumps({"result": False}))
            except Exception as e:
                logger.error(f"데이터 처리 중 오류 발생: {e}")
                await websocket.send_text(json.dumps({"result": False}))

    except Exception as e:
        logger.error(f"WebSocket 연결 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"WebSocket 연결 종료: {websocket.client}")

# face_detection 함수 수정
def face_detection(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.imread(image_path)

    if image is None or image.size == 0:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

    height, width, _ = image.shape

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark = face_landmarks.landmark

                nose_tip_y = landmark[4].y * height
                chin_tip_y = landmark[152].y * height
                forehead_tip_y = landmark[10].y * height

                vertical_diff_down = abs(nose_tip_y - chin_tip_y)
                vertical_diff_up = abs(nose_tip_y - forehead_tip_y)
                relative_tilt = vertical_diff_down / vertical_diff_up

                l_eyes_h = landmark[159].y * height
                l_eyes_r = landmark[145].y * height
                r_eyes_h = landmark[386].y * height
                r_eyes_r = landmark[374].y * height

                l_eyes = abs(l_eyes_h - l_eyes_r)
                r_eyes = abs(r_eyes_h - r_eyes_r)

                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]

                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                face_width = (max_x - min_x) * width
                face_height = (max_y - min_y) * height
                face_size = face_width * face_height

                result = {'far': 0, 'situation': 0}

                if face_size < 5000:
                    result['far'] = 2  # 멀다
                elif face_size > 20000:
                    result['far'] = 1  # 가깝다

                print(f'relative_tilt: {relative_tilt}, chin_tip_y: {chin_tip_y}, height: {height * 0.8}')
                if relative_tilt < 0.7 or relative_tilt > 3.43:
                    result['situation'] = 1  # 아래를 보고 있음

        print(f'result: {result}')
        return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
