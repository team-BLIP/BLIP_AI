from flask import Flask, request, jsonify
from image_face_pose_detection import main
import io
import cv2
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)

@app.route("/image", methods=['POST'])
def test():
    # JSON 데이터 가져오기
    params = request.get_json()

    # base64 문자열을 디코딩하여 이미지로 변환
    base64_image = params['image']  # base64로 인코딩된 이미지 데이터
    decoded_image = base64.b64decode(base64_image)  # base64를 디코딩하여 bytes로 변환

    # 디코딩된 이미지를 PIL 이미지로 로드
    img_out = Image.open(io.BytesIO(decoded_image))
    img_out = np.array(img_out)  # numpy 배열로 변환
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)  # RGB -> BGR로 변환
    cv2.imwrite('./image/img.jpg', img_out)  # 이미지 저장

    # main() 함수 호출
    res = main()

    # 응답 생성
    response = {
        'id': params['id'],
        "eyes": res["eyes"],
        "relative_tilt": res["relative_tilt"],
        'far': res["far"]
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
