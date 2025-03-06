import cv2
import pytesseract
import numpy as np

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract-OCR\tesseract.exe'

# 이미지 경로
IMAGE_PATH = r'/CV/face/image/hc/WIN_20250305_21_36_36_Pro.jpg'

selected_boxes = []  # 선택된 경계 상자 저장
selected_texts = []  # 선택된 단어 저장


def extract_word_boxes(image):
    """
    이미지에서 단어 단위로 경계 상자 추출
    """
    h, w, _ = image.shape
    data = pytesseract.image_to_data(image, lang='kor+eng', output_type=pytesseract.Output.DICT)
    word_boxes = []

    for i in range(len(data['text'])):
        if data['text'][i].strip():  # 단어가 존재할 때만 저장
            x1, y1, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            x2, y2 = x1 + width, y1 + height
            word_boxes.append((x1, y1, x2, y2, data['text'][i]))

    return word_boxes


def draw_word_boxes(image, boxes, selected):
    """
    이미지에 단어 경계 상자를 그림
    선택된 상자는 검정, 선택되지 않은 상자는 초록색
    """
    for (x1, y1, x2, y2, text) in boxes:
        color = (0, 0, 0) if (x1, y1, x2, y2) in selected else (0, 255, 0)  # 선택 여부에 따라 색 결정
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


def mouse_callback(event, x, y, flags, param):
    """
    마우스 클릭 이벤트 처리
    선택된 상자의 색을 검정으로 변경
    """
    global selected_boxes, selected_texts

    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 위치 (x, y)가 포함된 단어 상자를 찾음
        for (x1, y1, x2, y2, text) in param:
            if x1 <= x <= x2 and y1 <= y <= y2:
                # 선택된 상자와 텍스트 저장
                if (x1, y1, x2, y2) not in selected_boxes:
                    selected_boxes.append((x1, y1, x2, y2))
                    selected_texts.append(text)
                    print(f"선택된 단어: {text}")
                break


def main():
    global selected_boxes, selected_texts

    # 이미지 로드
    img = cv2.imdecode(np.fromfile(IMAGE_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 이미지 크기 확대 (x1.5 비율) -> 필요에 따라 변경 가능
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # 단어별 경계 상자 추출
    word_boxes = extract_word_boxes(img)

    # OpenCV 창 생성 및 전체화면 설정
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # 크기 조정 가능한 창
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # 전체화면 설정
    cv2.setMouseCallback("Image", mouse_callback, word_boxes)

    print("이미지 창에서 텍스트 경계 상자를 클릭하여 선택하세요. (종료하려면 'q' 키를 누르세요)")
    while True:
        # 표시를 위해 이미지를 새로 복사
        drawn_image = img.copy()
        draw_word_boxes(drawn_image, word_boxes, selected_boxes)

        # 이미지 표시
        cv2.imshow("Image", drawn_image)

        # 종료 키: q
        key = cv2.waitKey(1)
        if key == ord('q'):  # q 키
            break

    cv2.destroyAllWindows()

    # 선택된 텍스트 출력
    print("\n선택된 단어들:")
    print(" ".join(selected_texts))


if __name__ == "__main__":
    main()
