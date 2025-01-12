import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import os

def record_audio(duration, output_file):
    """
    음성을 녹음하고 MP3 파일로 저장합니다.

    :param duration: 녹음 시간(초)
    :param output_file: 저장할 MP3 파일 이름
    """
    # 녹음 설정
    sample_rate = 44100  # 44.1kHz (CD 품질)
    print("녹음을 시작합니다. {}초 동안 녹음합니다...".format(duration))
    
    # 녹음 실행
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # 녹음 종료까지 대기
    
    # WAV 파일로 임시 저장
    wav_file = os.path.join('Voice', 'temp.wav')
    write(wav_file, sample_rate, recording)
    print("녹음이 완료되었습니다. Voice 폴더에 WAV 파일로 저장되었습니다.")


record_audio(duration=10, output_file="output.mp3")
