import streamlit as st
import os
import json
import requests
import fal_client
import google.generativeai as genai
from google.cloud import texttospeech
import tempfile

# 👇 [이 부분 추가] Pillow 라이브러리 버전 호환성 패치
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# 👆 [여기까지 추가]

# 그 다음 moviepy를 불러와야 에러가 안 납니다.
from moviepy.editor import *

# 로컬 환경용 (.env 파일 로드)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- 1. 환경 및 UI 설정 ---
st.set_page_config(page_title="AI 영상 공장 (Final)", page_icon="🏭", layout="wide")
st.title("🏭 AI 유튜브 풀자동 공장 (설계도 완벽 구현)")

# 사이드바: 설정 및 캐릭터 앵커링
with st.sidebar:
    st.header("⚙️ 시스템 제어기")
    
    # API 키 검증
    api_status = {
        "Gemini (Brain)": "GOOGLE_API_KEY" in st.secrets or os.getenv("GOOGLE_API_KEY"),
        "Fal.ai (Visual)": "FAL_KEY" in st.secrets or os.getenv("FAL_KEY"),
        "Google TTS (Voice)": "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    }
    for name, ready in api_status.items():
        if ready: st.success(f"{name}: ON")
        else: st.error(f"{name}: OFF")
    
    st.divider()
    
    # [설계도 구현] Consistency Injection (일관성 주입)
    st.subheader("🖼️ 캐릭터 일관성 (Consistency)")
    default_char = "A young Korean office worker in a suit, simple clean lines, distinct facial features"
    character_desc = st.text_area("주인공 외모 묘사 (Anchor)", value=default_char, height=80)
    video_style = st.selectbox("화풍 (Style)", ["2D Webtoon Style", "Anime Style", "Realistic Cinematic"], index=0)
    
    num_scenes = st.slider("생성할 씬(Scene) 개수", 1, 5, 3)

# --- 2. 핵심 모듈 함수 ---

def get_api_key(key_name):
    if key_name in st.secrets: return st.secrets[key_name]
    return os.getenv(key_name)

def generate_script_json(topic, character_desc):
    """
    [Phase 1] Gemini: 기획 및 대본 작성 (언어 설정 강화)
    """
    api_key = get_api_key("GOOGLE_API_KEY")
    if not api_key: return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 프롬프트에 '언어 규칙(Language Rules)'을 명확히 추가했습니다.
        prompt = f"""
        Act as a professional YouTube content researcher and writer.
        Topic: '{topic}'
        
        [STRICT LANGUAGE RULES]
        1. "narrative": Must be in **KOREAN (한국어)** for the voiceover.
        2. "visual_prompt": Must be in **ENGLISH** for the image generator.
        
        [INSTRUCTION]
        1. FACT CHECK: Verify key facts about the topic.
        2. CHARACTER: You must include the character description in every visual prompt.
           If the provided 'character_desc' is in Korean, TRANSLATE it into English.
        3. STYLE: End every visual prompt with "{video_style}".
        
        [Ref: Character Description]
        "{character_desc}"
        
        Output ONLY valid JSON format:
        {{
          "video_title": "Catchy Title in Korean",
          "scenes": [
            {{ 
               "seq": 1, 
               "narrative": "한국어 내레이션 대본 (성우용)", 
               "visual_prompt": "Detailed English description starting with translated character traits, describing [Action/Background], ending with {video_style}" 
            }},
            ... (Total {num_scenes} scenes)
          ]
        }}
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        # 마크다운 제거
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
            
        return json.loads(text)
    except Exception as e:
        st.error(f"🧠 Gemini 기획 오류: {e}")
        return None

def generate_image(prompt, filename):
    """
    [Phase 2] Nano Banana (Fal.ai): 이미지 생성
    """
    output_path = os.path.join(tempfile.gettempdir(), filename)
    os.environ["FAL_KEY"] = get_api_key("FAL_KEY")
    
    try:
        handler = fal_client.submit(
            "fal-ai/flux/dev",
            arguments={
                "prompt": prompt,
                "image_size": "landscape_16_9",
                "seed": 42, # [설계도 구현] Seed 고정으로 일관성 확보
                "num_inference_steps": 25
            }
        )
        image_url = handler.get()['images'][0]['url']
        with open(output_path, 'wb') as f:
            f.write(requests.get(image_url).content)
        return output_path
    except Exception as e:
        st.error(f"🎨 이미지 생성 오류: {e}")
        return None

def generate_audio(text, filename):
    """
    [Phase 2] Google TTS: 오디오 생성
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
    """
    [Phase 3] Zoom Effect: 밋밋한 정지 이미지를 동적으로 변환 (Ken Burns 효과)
    """
    def effect(get_frame, t):
        img = get_frame(t)
        h, w = img.shape[:2]
        # 시간이 지날수록 조금씩 확대 (Zoom In)
        scale = 1 + zoom_ratio * t
        
        # OpenCV로 리사이즈 (속도 최적화)
        import cv2
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 중앙 크롭 (Center Crop)
        x = (new_w - w) // 2
        y = (new_h - h) // 2
        return img_resized[y:y+h, x:x+w]

    return clip.fl(effect)

# --- 3. 메인 실행 컨트롤러 ---

topic = st.text_input("영상 주제 (Topic)", placeholder="예: 비트코인 1억 돌파의 이유")

if st.button("🚀 시스템 가동 (Start Process)", type="primary"):
    if not topic: st.stop()
    
    status_box = st.status("🏗️ 공장 가동 중...", expanded=True)
    
    # --- Phase 1: 기획 ---
    status_box.write("🧠 Phase 1: Gemini가 시나리오를 설계 중입니다...")
    script_data = generate_script_json(topic, character_desc)
    
    if not script_data:
        status_box.update(label="❌ 기획 단계 실패", state="error")
        st.stop()
        
    scenes = script_data.get("scenes", [])
    st.json(script_data) # 기획안 출력
    
    # --- Phase 2: 자산 생성 ---
    status_box.write("🎨 Phase 2: 이미지와 음성을 병렬(순차) 생성 중입니다...")
    progress_bar = st.progress(0)
    generated_clips = []
    
    for i, scene in enumerate(scenes):
        idx = scene['seq']
        
        # 파일명 생성
        img_name = f"img_{idx}.png"
        aud_name = f"aud_{idx}.mp3"
        
        # 생성 (에러 시 해당 씬 스킵)
        img_path = generate_image(scene['visual_prompt'], img_name)
        aud_path = generate_audio(scene['narrative'], aud_name)
        
        if img_path and aud_path:
            # 클립 조합
            try:
                audio_clip = AudioFileClip(aud_path)
                # [설계도 구현] 기본 이미지 클립 생성
                img_clip = ImageClip(img_path).set_duration(audio_clip.duration).resize(height=720)
                
                # [설계도 구현] Zoom Effect 적용! (정지 화면 탈피)
                # *주의: Streamlit Cloud 사양상 복잡하면 느려질 수 있어 가볍게 적용
                video_clip = create_zoom_effect(img_clip, zoom_ratio=0.04) 
                
                video_clip = video_clip.set_audio(audio_clip)
                generated_clips.append(video_clip)
                
                # 중간 결과 보여주기
                with st.expander(f"Scene {idx} 미리보기", expanded=False):
                    st.image(img_path, caption=f"Scene {idx}")
                    st.audio(aud_path)
                    
            except Exception as e:
                st.warning(f"Scene {idx} 조립 중 오류: {e}")
        
        progress_bar.progress((i + 1) / len(scenes))
        
    # --- Phase 3: 최종 조립 ---
    if generated_clips:
        status_box.write("🎬 Phase 3: FFmpeg 엔진으로 영상을 렌더링 중입니다...")
        try:
            # 안전한 합치기
            final_video = concatenate_videoclips(generated_clips, method="compose")
            
            # 파일명 안전하게 변환
            safe_title = "".join([c for c in script_data['video_title'] if c.isalnum()]).strip() or "output"
            output_path = os.path.join(tempfile.gettempdir(), f"{safe_title}.mp4")
            
            # 렌더링 (속도 우선)
            final_video.write_videofile(
                output_path, 
                fps=24, 
                codec='libx264', 
                audio_codec='aac', 
                preset='ultrafast'
            )
            
            status_box.update(label="✅ 모든 공정 완료!", state="complete", expanded=False)
            st.success("🎉 영상 완성!")
            
            st.video(output_path)
            with open(output_path, 'rb') as f:
                st.download_button("📥 영상 다운로드", f, file_name=f"{safe_title}.mp4")
                
        except Exception as e:
            st.error(f"렌더링 오류: {e}")
    else:
        st.error("❌ 생성된 클립이 없습니다.")