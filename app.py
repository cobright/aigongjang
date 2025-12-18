import streamlit as st
import os
import json
import requests
import google.generativeai as genai_old # 텍스트용 (구버전 SDK)
from google import genai # ⭐ 이미지용 (신버전 SDK)
from google.genai import types
from google.cloud import texttospeech
import tempfile
from moviepy.editor import *
import mimetypes

# 로컬 환경용 (.env 파일 로드)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- 1. 환경 및 UI 설정 ---
st.set_page_config(page_title="AI 영상 공장 (Google Edition)", page_icon="🍌", layout="wide")
st.title("🍌 AI 영상 공장 (Gemini 3 Pro Image)")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 시스템 제어기")
    
    # API 키 검증 (GEMINI_API_KEY는 구글 AI 스튜디오 키와 동일하게 사용)
    api_status = {
        "Gemini API (Text & Image)": "GOOGLE_API_KEY" in st.secrets or os.getenv("GOOGLE_API_KEY"),
        "Google TTS (Voice)": "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    }
    for name, ready in api_status.items():
        if ready: st.success(f"{name}: ON")
        else: st.error(f"{name}: OFF")
    
    st.divider()
    
    st.subheader("🖼️ 캐릭터 일관성 (Consistency)")
    default_char = "A young Korean office worker in a suit, simple clean lines, distinct facial features"
    character_desc = st.text_area("주인공 외모 묘사 (Anchor)", value=default_char, height=80)
    video_style = st.selectbox("화풍 (Style)", ["2D Webtoon Style", "Anime Style", "Realistic Cinematic"], index=0)
    
    num_scenes = st.slider("생성할 씬(Scene) 개수", 1, 5, 2)

# --- 2. 핵심 모듈 함수 ---

def get_api_key(key_name):
    if key_name in st.secrets: return st.secrets[key_name]
    return os.getenv(key_name)

def generate_script_json(topic, character_desc):
    """
    [Text] Gemini (Old SDK): 기획 및 대본 작성
    """
    api_key = get_api_key("GOOGLE_API_KEY")
    if not api_key: return None
    
    try:
        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Act as a professional YouTube content researcher and writer.
        Topic: '{topic}'
        
        [STRICT LANGUAGE RULES]
        1. "narrative": Must be in **KOREAN (한국어)** for the voiceover.
        2. "visual_prompt": Must be in **ENGLISH** for the image generator.
        
        [INSTRUCTION]
        1. FACT CHECK: Verify key facts about the topic.
        2. CHARACTER: You must include the character description in every visual prompt.
        3. STYLE: End every visual prompt with "{video_style}".
        
        [Ref: Character Description]
        "{character_desc}"
        
        Output ONLY valid JSON format:
        {{
          "video_title": "Catchy Title",
          "scenes": [
            {{ 
               "seq": 1, 
               "narrative": "한국어 내레이션", 
               "visual_prompt": "English visual description..." 
            }},
            ... (Total {num_scenes} scenes)
          ]
        }}
        """
        response = model.generate_content(prompt)
        text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        st.error(f"🧠 Gemini 기획 오류: {e}")
        return None

def generate_image_google(prompt, filename):
    """
    [Image] Gemini (New SDK): 이미지 생성 (사용자가 제공한 코드 적용)
    """
    # 임시 파일 경로 설정
    output_path = os.path.join(tempfile.gettempdir(), filename)
    
    # 1. 클라이언트 초기화 (GOOGLE_API_KEY 사용)
    api_key = get_api_key("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Google API Key가 없습니다.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        
        # 2. 모델 및 설정 (사용자 제공 코드 기반)
        # 주의: 'gemini-3-pro-image-preview' 모델 접근 권한이 없으면 'imagen-3.0-generate-001' 등을 써야 할 수 있습니다.
        model_id = "gemini-3-pro-image-preview" 

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"], # 텍스트 없이 이미지만 요청
            image_config=types.ImageConfig(image_size="1K"),
        )

        # 3. 생성 요청 및 바이너리 저장
        # 스트리밍 방식 대신 동기 호출로 처리하여 파일 저장
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=generate_content_config,
        )
        
        # 응답 처리
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    # 이미지 데이터 추출
                    img_data = part.inline_data.data
                    
                    # 파일 저장
                    with open(output_path, "wb") as f:
                        f.write(img_data)
                    
                    return output_path
        
        st.error("❌ 구글 이미지 생성 실패: 이미지가 반환되지 않았습니다.")
        return None

    except Exception as e:
        st.error(f"🎨 Google 이미지 생성 오류: {e}")
        # 혹시 모델명 때문에 에러가 난다면 팁을 줍니다.
        if "404" in str(e) or "Not Found" in str(e):
             st.warning("⚠️ 팁: 'gemini-3-pro-image-preview' 모델 접근 권한이 없을 수 있습니다. 코드를 'imagen-3.0-generate-001'로 변경해보세요.")
        return None

def generate_audio(text, filename):
    """
    [Voice] Google TTS: 오디오 생성
    """
    output_path = os.path.join(tempfile.gettempdir(), filename)
    creds_json = get_api_key("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds_json: return None
    
    try:
        from google.oauth2 import service_account
        creds_info = json.loads(creds_json, strict=False)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
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
        return clip # cv2 없으면 효과 없이 원본 리턴

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

topic = st.text_input("영상 주제 (Topic)", placeholder="예: 구글 제미나이가 이미지를 그리는 법")

if st.button("🚀 Google 시스템 가동", type="primary"):
    if not topic: st.stop()
    
    status_box = st.status("🏗️ 공장 가동 중...", expanded=True)
    
    # --- Phase 1: 기획 ---
    status_box.write("🧠 Phase 1: Gemini가 시나리오를 설계 중입니다...")
    script_data = generate_script_json(topic, character_desc)
    
    if not script_data:
        status_box.update(label="❌ 기획 단계 실패", state="error")
        st.stop()
        
    scenes = script_data.get("scenes", [])
    st.json(script_data)
    
    # --- Phase 2: 자산 생성 ---
    status_box.write("🎨 Phase 2: Google Image & Voice 생성 중...")
    progress_bar = st.progress(0)
    generated_clips = []
    
    for i, scene in enumerate(scenes):
        idx = scene['seq']
        img_name = f"img_{idx}.png"
        aud_name = f"aud_{idx}.mp3"
        
        # ⭐ 여기가 변경된 부분: generate_image_google 사용
        img_path = generate_image_google(scene['visual_prompt'], img_name)
        aud_path = generate_audio(scene['narrative'], aud_name)
        
        if img_path and aud_path:
            try:
                audio_clip = AudioFileClip(aud_path)
                img_clip = ImageClip(img_path).set_duration(audio_clip.duration).resize(height=720)
                
                # 줌 효과 적용 (OpenCV 설치된 경우만)
                video_clip = create_zoom_effect(img_clip) 
                
                video_clip = video_clip.set_audio(audio_clip)
                generated_clips.append(video_clip)
                
                with st.expander(f"Scene {idx} 미리보기", expanded=False):
                    st.image(img_path, caption=f"Google Image - Scene {idx}")
                    st.audio(aud_path)
                    
            except Exception as e:
                st.warning(f"Scene {idx} 조립 중 오류: {e}")
        
        progress_bar.progress((i + 1) / len(scenes))
        
    # --- Phase 3: 최종 조립 ---
    if generated_clips:
        status_box.write("🎬 Phase 3: 영상 렌더링 중...")
        try:
            final_video = concatenate_videoclips(generated_clips, method="compose")
            safe_title = "".join([c for c in script_data['video_title'] if c.isalnum()]).strip() or "output"
            output_path = os.path.join(tempfile.gettempdir(), f"{safe_title}.mp4")
            
            final_video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', preset='ultrafast')
            
            status_box.update(label="✅ 완료!", state="complete", expanded=False)
            st.success("🎉 구글 생성 영상 완성!")
            st.video(output_path)
            
        except Exception as e:
            st.error(f"렌더링 오류: {e}")
    else:
        st.error("❌ 생성된 클립이 없습니다.")