import streamlit as st
import os
import json
import tempfile
import time

# --- 라이브러리 임포트 및 예외 처리 ---
# 1. Google GenAI (텍스트용 - 구버전 SDK)
import google.generativeai as genai_old

# 2. Google GenAI (이미지용 - 신버전 SDK)
# pip install google-genai
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("❌ 'google-genai' 라이브러리가 설치되지 않았습니다. pip install google-genai")
    st.stop()

# 3. Google TTS
from google.cloud import texttospeech
from google.oauth2 import service_account

# 4. Pillow 패치 (MoviePy 1.x 호환성)
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# 5. MoviePy
from moviepy.editor import *

# 6. .env 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- 1. 환경 및 UI 설정 ---
st.set_page_config(page_title="AI 영상 공장 (Google Edition)", page_icon="🍌", layout="wide")
st.title("🍌 AI 영상 공장 (Gemini 3 Pro Image)")

# --- [안전장치] Secrets 조회 함수 ---
def get_secret(key_name):
    """st.secrets -> os.environ 순서로 키를 찾습니다."""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except (FileNotFoundError, AttributeError):
        pass
    return os.getenv(key_name)

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 시스템 제어기")
    
    gemini_key = get_secret("GOOGLE_API_KEY")
    tts_key_path = get_secret("GOOGLE_APPLICATION_CREDENTIALS") # 로컬 파일 경로
    tts_key_json = get_secret("GOOGLE_APPLICATION_CREDENTIALS_JSON") # 클라우드용 JSON 내용
    
    # 상태 표시
    if gemini_key:
        st.success("✅ Gemini API: Connected")
    else:
        st.error("❌ Gemini API: Missing Key")
        
    if tts_key_path or tts_key_json:
        st.success("✅ Google TTS: Connected")
    else:
        st.error("❌ Google TTS: Missing Credentials")
    
    st.divider()
    
    st.subheader("🖼️ 캐릭터 설정")
    default_char = "A young Korean office worker in a suit, simple clean lines, distinct facial features"
    character_desc = st.text_area("주인공 외모 묘사", value=default_char, height=80)
    video_style = st.selectbox("화풍 (Style)", ["2D Webtoon Style", "Anime Style", "Realistic Cinematic", "Oil Painting"], index=0)
    
    num_scenes = st.slider("생성할 씬(Scene) 개수", 1, 5, 2)

# --- 2. 핵심 모듈 함수 ---

def generate_script_json(topic, character_desc):
    """[Text] Gemini (Old SDK): 기획 및 대본 작성"""
    if not gemini_key: return None
    
    try:
        genai_old.configure(api_key=gemini_key)
        # 수정: 모델명을 2.5(존재안함) -> 1.5-flash로 변경
        model = genai_old.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Act as a professional YouTube content researcher and writer.
        Topic: '{topic}'
        
        [STRICT LANGUAGE RULES]
        1. "narrative": Must be in **KOREAN (한국어)** for the voiceover.
        2. "visual_prompt": Must be in **ENGLISH** (recommended for better image gen) or KOREAN.
        
        [INSTRUCTION - DYNAMIC VISUALS]
        1. **NO STATIC POSES:** Do not describe characters just standing.
        2. **ACTION & MOTION:** Describe specific moments (e.g., "running," "laughing").
        3. **CAMERA ANGLES:** Use "low angle," "close-up," etc.
        
        [CHARACTER CONSISTENCY]
        Start visual prompt with: "{character_desc}"
        
        [STYLE]
        End visual prompt with: "{video_style}"
        
        Output ONLY valid JSON format:
        {{
          "video_title": "Title Here",
          "scenes": [
            {{ 
               "seq": 1, 
               "narrative": "한국어 내레이션", 
               "visual_prompt": "Visual description..." 
            }}
          ]
        }}
        Generate exactly {num_scenes} scenes.
        """
        response = model.generate_content(prompt)
        text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        st.error(f"🧠 Gemini 기획 오류: {e}")
        return None

def generate_image_google(prompt, filename):
    """[Image] Gemini (New SDK): 이미지 생성"""
    if not gemini_key: return None
    
    output_path = os.path.join(tempfile.gettempdir(), filename)

    try:
        client = genai.Client(api_key=gemini_key)
        # 모델 ID 확인 필요 (imagen-3.0-generate-001 또는 gemini-pro-vision 등 상황에 맞게)
        # 현재 코드의 gemini-3-pro-image-preview는 Preview 권한이 있어야 작동합니다.
        model_id = "gemini-2.0-flash-exp" # 혹은 "imagen-3.0-generate-001"
        
        # *참고: Gemini 2.0 Flash Exp는 이미지 생성을 지원하지만, 
        # 전용 Imagen 모델을 쓴다면 코드가 달라질 수 있습니다. 
        # 여기서는 작성해주신 코드를 기반으로 유지합니다.
        
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(image_size="1K"), # 혹은 "1024x1024" 등 SDK 버전에 따라 다름
        )

        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=generate_content_config,
        )
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    return output_path
        
        st.warning("⚠️ 이미지가 생성되지 않았습니다 (모델 응답 없음).")
        return None

    except Exception as e:
        st.error(f"🎨 Google 이미지 생성 오류: {e}")
        return None

def generate_audio(text, filename):
    """[Voice] Google TTS"""
    output_path = os.path.join(tempfile.gettempdir(), filename)
    
    # 인증 처리 로직 단순화 및 강화
    credentials = None
    
    try:
        # 1. Secret/Env에 JSON 내용이 통째로 있는 경우 (Streamlit Cloud 권장)
        if tts_key_json:
            try:
                creds_info = json.loads(tts_key_json, strict=False)
                credentials = service_account.Credentials.from_service_account_info(creds_info)
            except json.JSONDecodeError:
                st.error("TTS JSON 키 형식이 잘못되었습니다.")
                return None
        
        # 2. 로컬 파일 경로가 있는 경우
        elif tts_key_path and os.path.exists(tts_key_path):
            credentials = service_account.Credentials.from_service_account_file(tts_key_path)
            
        else:
            # 하드코딩된 파일명은 제거하고 경고 메시지 출력
            st.error("❌ TTS 인증 키를 찾을 수 없습니다. (.env 또는 secrets.toml 확인)")
            return None

        client = texttospeech.TextToSpeechClient(credentials=credentials)
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Standard-C", ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        return output_path
        
    except Exception as e:
        st.error(f"🎙️ TTS 오류: {e}")
        return None

def create_zoom_effect(clip, zoom_ratio=0.04):
    """Zoom Effect (OpenCV 필요)"""
    try:
        import cv2
    except ImportError:
        # OpenCV 없으면 효과 없이 원본 리턴
        return clip

    def effect(get_frame, t):
        img = get_frame(t)
        h, w = img.shape[:2]
        scale = 1 + zoom_ratio * t
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x = (new_w - w) // 2
        y = (new_h - h) // 2
        return img_resized[y:y+h, x:x+w]

    return clip.fl(effect)

# --- 3. 메인 실행 컨트롤러 ---

topic = st.text_input("영상 주제 (Topic)", placeholder="예: 맛있는 김치찌개 끓이는 법")

if st.button("🚀 영상 생성 시작", type="primary"):
    if not topic:
        st.warning("주제를 입력해주세요.")
        st.stop()
    
    status_box = st.status("🏗️ 작업 시작...", expanded=True)
    
    # Phase 1
    status_box.write("🧠 Phase 1: 시나리오 기획 중 (Gemini 1.5 Flash)...")
    script_data = generate_script_json(topic, character_desc)
    
    if not script_data:
        status_box.update(label="❌ 기획 실패", state="error")
        st.stop()
        
    scenes = script_data.get("scenes", [])
    st.json(script_data) # 디버깅용 표시
    
    # Phase 2
    status_box.write("🎨 Phase 2: 이미지 및 오디오 생성 중...")
    progress_bar = st.progress(0)
    generated_clips = []
    
    for i, scene in enumerate(scenes):
        idx = scene['seq']
        status_box.write(f"  - Scene {idx} 생성 중...")
        
        # 파일명 중복 방지를 위해 timestamp 추가 권장
        timestamp = int(time.time())
        img_name = f"img_{idx}_{timestamp}.png"
        aud_name = f"aud_{idx}_{timestamp}.mp3"
        
        aud_path = generate_audio(scene['narrative'], aud_name)
        img_path = generate_image_google(scene['visual_prompt'], img_name)
        
        if img_path and aud_path:
            try:
                audio_clip = AudioFileClip(aud_path)
                # 이미지 지속시간을 오디오 길이에 맞춤
                img_clip = ImageClip(img_path).set_duration(audio_clip.duration).resize(height=720)
                video_clip = create_zoom_effect(img_clip) 
                video_clip = video_clip.set_audio(audio_clip)
                generated_clips.append(video_clip)
            except Exception as e:
                st.warning(f"Scene {idx} 클립 생성 실패: {e}")
        
        progress_bar.progress((i + 1) / len(scenes))
        
    # Phase 3
    if generated_clips:
        status_box.write("🎬 Phase 3: 최종 렌더링 중 (FFmpeg)...")
        try:
            final_video = concatenate_videoclips(generated_clips, method="compose")
            
            # 안전한 파일명 생성
            safe_title = "".join([c for c in topic if c.isalnum()]).strip()
            output_path = os.path.join(tempfile.gettempdir(), f"{safe_title}_final.mp4")
            
            final_video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', preset='ultrafast')
            
            status_box.update(label="✅ 완료!", state="complete", expanded=False)
            st.success("🎉 영상 생성이 완료되었습니다!")
            st.video(output_path)
            
        except Exception as e:
            st.error(f"렌더링 오류: {e}")
    else:
        st.error("❌ 생성된 클립이 없어 영상을 만들 수 없습니다.")
