import whisper
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env에서 파일 경로 가져오기
audio_file_path = os.getenv("AUDIO_FILE_PATH")

# 모델 로드
model = whisper.load_model("medium")

# 오디오 파일을 텍스트로 변환
result = model.transcribe(audio_file_path, language="ko")

# 변환된 텍스트 출력
print(result["text"])
