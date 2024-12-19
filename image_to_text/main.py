import cv2
from matplotlib import pyplot as plt
import pytesseract
import numpy as np

# pytesseract 사용을 위한 코드 추가
pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract-OCR\tesseract.exe'

# 사용할 이미지 위치 주소 변수에 저장
image = r"C:\폴더들\project\AIexpo\img\image.png"

# 이미지 파일 읽기
img = cv2.imdecode(np.fromfile(r'C:\폴더들\project\AIexpo\img\image.png', dtype=np.uint8), cv2.IMREAD_COLOR)


# 글자 추출 전처리를 위해 흑백으로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 흑백 처리한 이미지 확인
plt.imshow(gray)
plt.show()

# 한글과 영문 추출을 위한 config 내용 변수에 저장
config = ('-l kor+eng --oem 3 --psm 11')

# 이미지에서 글자 추출
output = pytesseract.image_to_string(gray, config=config)

# 추출한 글자 확인
print(output)