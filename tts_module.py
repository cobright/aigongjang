# tts_module.py
import os
from google.cloud import texttospeech
from dotenv import load_dotenv

load_dotenv()
# GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 자동으로 로드되어 인증에 사용됩니다.

def generate_audio(text, filename, output_dir="assets/audio"):
    """
    텍스트를 받아 음성 파일을 생성하고 지정된 경로에 저장하는 함수
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"⏭️ 오디오 스킵: {filename} (이미 존재함)")
        return filepath

    print(f"🎙️ 구글 TTS: 음성 생성 중... ({filename})")
    
    try:
        # 1. 클라이언트 생성
        client = texttospeech.TextToSpeechClient()

        # 2. 입력 텍스트 설정
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # 3. 목소리 설정 (한국어, 남성 표준 음성 예시)
        # *참고: 구글 클라우드 콘솔에서 원하는 목소리 ID 확인 가능
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Standard-C", 
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )

        # 4. 오디오 설정 (MP3)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # 5. 음성 합성 요청
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # 6. 파일 저장
        with open(filepath, "wb") as out:
            out.write(response.audio_content)
            print(f"✅ 오디오 저장 완료: {filepath}")
            return filepath
            
    except Exception as e:
        print(f"❌ 구글 TTS 오류: {e}")
        return None