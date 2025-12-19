import streamlit as st
import os
import json
import tempfile
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Pillow 모듈

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
    
    num_scenes = st.slider("생성할 씬(Scene) 개수 (권장: 4개 이상)", 2, 6, 4)

    st.divider()
    st.subheader("🎵 배경음악 (BGM)")
    bgm_mood = st.selectbox(
        "분위기 선택", 
        ["🌞 밝고 경쾌한 (Bright)", "☕ 차분한 (Calm)", "🔥 박진감 넘치는 (Epic)", "🔇 없음 (Mute)"],
        index=0
    )

# --- 2. 핵심 모듈 함수 ---

def generate_script_json(topic, character_desc, num_scenes):
    """
    [Text] Gemini (Old SDK): 
    기존의 단순 생성 방식을 버리고, 'Hook-Body-Conclusion' 구조를 강제합니다.
    """
    if not gemini_key: return None
    
    try:
        genai_old.configure(api_key=gemini_key)
        
        # 요청하신 대로 2.5 버전 사용 (주의: 2.5가 아직 배포 전이라면 1.5나 2.0-flash-exp 사용)
        model = genai_old.GenerativeModel('gemini-2.5-flash') 
        
        # 구조화된 프롬프트 설계
        prompt = f"""
        You are an expert Content Creator for viral YouTube Shorts and TikTok.
        Your goal is to create a script for the topic: '{topic}'
        
        [STRUCTURE STRATEGY - MUST FOLLOW]
        Organize the {num_scenes} scenes strictly according to this flow:
        1. **HOOK**: Start with a question or shock.
        2. **BODY**: Explain the core story.
        3. **OUTRO**: Conclusion and CTA.

        [STRICT LANGUAGE RULES]
        1. "narrative": Must be in **KOREAN (한국어)**. Conversational style.
        2. "visual_prompt": Must be in **KOREAN (한국어)**.
        
        [DYNAMIC VISUAL INSTRUCTION - IMPORTANT]
        To make the video dynamic, **you MUST provide 2 or 3 different visual descriptions per scene**, separated by " || ".
        - Example: "Close up of a man eating kimchi || Wide shot of the restaurant || The man giving a thumbs up"
        - This will generate 3 images shown in sequence for this one scene.
        
        [OUTPUT FORMAT]
        Return ONLY valid JSON:
        {{
          "video_title": "Title in Korean",
          "scenes": [
            {{ 
               "seq": 1, 
               "section": "HOOK",
               "narrative": "Korean voiceover...", 
               "visual_prompt": "Image Desc 1 || Image Desc 2" 
            }},
            ...
          ]
        }}
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # 마크다운 제거 (안전장치)
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
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
        model_id = "gemini-3-pro-image-preview" # 혹은 "imagen-3.0-generate-001"
        
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

def get_bgm_path(mood):
    """
    선택된 분위기(mood)에 맞는 무료 BGM을 다운로드하여 경로를 반환합니다.
    (저작권 무료 음원: Kevin MacLeod 등 Open Source 활용)
    """
    if mood == "🔇 없음 (Mute)":
        return None
        
    # 분위기별 음원 URL (안정적인 GitHub Raw 소스 등을 사용하는 것이 좋으나, 예시로 무료 음원 링크 사용)
    # 실제 운영 시에는 본인의 서버나 S3 링크로 교체하는 것을 권장합니다.
    bgm_urls = {
        "🌞 밝고 경쾌한 (Bright)": "https://www.bensound.com/bensound-music/bensound-ukulele.mp3", # 예시 URL
        "☕ 차분한 (Calm)": "https://www.bensound.com/bensound-music/bensound-slowmotion.mp3", 
        "🔥 박진감 넘치는 (Epic)": "https://www.bensound.com/bensound-music/bensound-evolution.mp3"
    }
    
    url = bgm_urls.get(mood)
    if not url: return None
    
    # 임시 파일명 (분위기별로 캐싱)
    filename = f"bgm_{mood[:2]}.mp3"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    
    # 이미 다운로드 받은 파일이 있으면 재사용 (속도 향상)
    if os.path.exists(filepath):
        return filepath
        
    try:
        # User-Agent 헤더 추가 (차단 방지)
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    except Exception as e:
        st.warning(f"BGM 다운로드 실패: {e}")
        return None


def get_korean_font():
    """
    한글 폰트(나눔고딕)를 임시 폴더에 다운로드하여 경로를 반환합니다.
    (한글 깨짐 방지용)
    """
    font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf"
    font_path = os.path.join(tempfile.gettempdir(), "NanumGothic-Bold.ttf")
    
    if not os.path.exists(font_path):
        try:
            response = requests.get(font_url)
            with open(font_path, "wb") as f:
                f.write(response.content)
        except Exception:
            return None # 다운로드 실패 시 기본 폰트 시도
            
    return font_path

def create_subtitle(text, duration, font_path):
    """
    [무설치 버전] Pillow를 사용하여 자막 이미지를 생성합니다.
    ImageMagick이 필요 없어 오류가 나지 않습니다.
    """
    try:
        # 1. 캔버스 크기 설정 (HD 해상도 기준)
        w, h = 1280, 720
        # 투명 배경 이미지 생성 (RGBA)
        img = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # 2. 폰트 로드
        font_size = 50
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                # 폰트 경로 없으면 기본 폰트 (한글 깨질 수 있음)
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # 3. 텍스트 줄바꿈 처리 (간단 버전)
        if len(text) > 25:
            text = text[:25] + "\n" + text[25:]
        
        # 4. 텍스트 크기 계산 및 위치 잡기 (중앙 하단)
        # textbbox는 Pillow 최신 버전 기준
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
        
        x = (w - text_w) / 2
        y = h - 100 # 밑에서 100px 위
        
        # 5. 글자 테두리(Stroke) 그리기 (검은색)
        stroke_width = 3
        draw.text((x, y), text, font=font, fill="white", stroke_width=stroke_width, stroke_fill="black")

        # 6. MoviePy 클립으로 변환
        # Pillow 이미지 -> Numpy 배열 -> ImageClip
        return ImageClip(np.array(img)).set_duration(duration)

    except Exception as e:
        st.error(f"자막 생성 중 오류 발생: {e}")
        return None


# --- 3. 메인 실행 컨트롤러 ---

# 세션 상태 초기화 (새로고침 해도 데이터 유지)
if "script_data" not in st.session_state:
    st.session_state["script_data"] = None
if "step" not in st.session_state:
    st.session_state["step"] = 1

st.divider()
st.header("Step 1. 기획안 작성")
topic = st.text_input("영상 주제 (Topic)", placeholder="예: 집에서 만드는 스타벅스 돌체라떼 레시피")

# [버튼 1] 기획안 생성
if st.button("💡 1. 기획안(대본) 생성하기", type="primary", use_container_width=True):
    if not topic:
        st.warning("주제를 입력해주세요.")
        st.stop()
        
    with st.spinner("🧠 Gemini가 기승전결(Hook-Body-CTA) 구조로 기획 중입니다..."):
        # 1단계에서 만든 구조화된 함수 호출
        script_data = generate_script_json(topic, character_desc, num_scenes)
        
        if script_data:
            st.session_state["script_data"] = script_data
            st.session_state["step"] = 2
            st.rerun() # 화면 갱신
        else:
            st.error("기획안 생성에 실패했습니다. 다시 시도해주세요.")

# [UI] 대본 수정 및 확정 단계
if st.session_state["step"] >= 2 and st.session_state["script_data"]:
    st.divider()
    st.header("Step 2. 대본 및 연출 수정")
    st.info("💡 꿀팁: '그림 묘사' 칸에 ` || ` 기호를 넣으면 컷이 쪼개집니다. (예: 남자의 얼굴 || 놀라는 표정)")

    data = st.session_state["script_data"]
    scenes = data.get("scenes", [])
    
    # 폼(Form)을 사용하여 한 번에 데이터 수집
    with st.form("script_edit_form"):
        # 타이틀 수정
        new_title = st.text_input("영상 제목", value=data.get("video_title", ""))
        
        edited_scenes = []
        for i, scene in enumerate(scenes):
            st.subheader(f"🎬 Scene {scene['seq']} ({scene.get('section', 'General')})")
            
            # 레이아웃 2단 분리 (왼쪽: 대본 / 오른쪽: 그림 묘사)
            col1, col2 = st.columns(2)
            
            with col1:
                # key를 고유하게 지정해야 함
                new_narrative = st.text_area(
                    label="🗣️ 내레이션 (한국어)", 
                    value=scene['narrative'], 
                    height=100,
                    key=f"narr_area_{i}"
                )
            
            with col2:
                new_visual = st.text_area(
                    label="🖼️ 그림 묘사 (영어 권장)", 
                    value=scene['visual_prompt'], 
                    height=100,
                    key=f"vis_area_{i}"
                )
            
            # 수정된 내용을 리스트에 담을 준비 (실제 업데이트는 버튼 클릭 시 처리)
            # 여기서는 폼 내부라 UI만 보여주고, 데이터 처리는 아래 submit 버튼 이후에 함.
            
        # [버튼 2] 영상 생성 시작
        generate_btn = st.form_submit_button("🎬 2. 이 내용으로 영상 만들기 (Start Generation)", type="primary", use_container_width=True)

    # 폼 제출 버튼이 눌렸을 때 실행
    if generate_btn:
        # 수정된 데이터 수집
        final_scenes = []
        for i, org_scene in enumerate(scenes):
            final_scenes.append({
                "seq": org_scene['seq'],
                "narrative": st.session_state[f"narr_area_{i}"],      # 위 text_area의 key값으로 읽어옴
                "visual_prompt": st.session_state[f"vis_area_{i}"]   # 위 text_area의 key값으로 읽어옴
            })
            
        # 본격적인 생성 시작
        status_box = st.status("🏗️ 영상 제작 공장 가동 중...", expanded=True)
        
        # --- Phase 2: 다이나믹 컷 생성 (컷 쪼개기 적용) ---
        status_box.write("🎨 Phase 2: 다이나믹 컷(이미지 분할) 및 오디오 생성 중...")
        progress_bar = st.progress(0)
        generated_clips = []
        
        for i, scene in enumerate(final_scenes):
            idx = scene['seq']
            status_box.write(f"  - Scene {idx} 작업 중...")
            
            timestamp = int(time.time())
            
            # 1. 오디오 먼저 생성 (길이를 알아야 컷을 나눌 수 있음)
            aud_name = f"aud_{idx}_{timestamp}.mp3"
            aud_path = generate_audio(scene['narrative'], aud_name)
            
            if not aud_path:
                continue # 오디오 실패 시 건너뜀
                
            audio_clip = AudioFileClip(aud_path)
            scene_duration = audio_clip.duration
            
            # 2. 비주얼 프롬프트 분석 ('||' 기준으로 쪼개기)
            # 예: "고양이가 밥먹음 || 고양이가 잠" -> ["고양이가 밥먹음", "고양이가 잠"]
            raw_prompts = scene['visual_prompt'].split('||')
            visual_prompts = [p.strip() for p in raw_prompts if p.strip()]
            
            # 만약 쪼갤 게 없으면 그냥 1개로 처리
            if not visual_prompts:
                visual_prompts = [scene['visual_prompt']]
            
            # 컷 당 지속 시간 계산 (총 시간 / 이미지 개수)
            # 예: 오디오 6초, 이미지 3장이면 -> 각 2초씩 보여줌
            clip_duration = scene_duration / len(visual_prompts)
            
            scene_sub_clips = [] # 이 씬을 구성할 작은 조각 영상들
            
            # 3. 각 컷 별로 이미지 생성 및 클립 만들기
            for sub_idx, prompt in enumerate(visual_prompts):
                img_name = f"img_{idx}_{sub_idx}_{timestamp}.png"
                status_box.write(f"    └ 컷 {sub_idx+1}/{len(visual_prompts)}: {prompt[:20]}...")
                
                img_path = generate_image_google(prompt, img_name)
                
                if img_path:
                    try:
                        # 이미지 클립 생성 (계산된 시간만큼)
                        sub_clip = ImageClip(img_path).set_duration(clip_duration).resize(height=720)
                        
                        # 줌 효과도 각각 적용 (더 역동적임)
                        sub_clip = create_zoom_effect(sub_clip)
                        scene_sub_clips.append(sub_clip)
                    except Exception as e:
                        st.warning(f"이미지 처리 중 오류: {e}")
            
            # 4. 조각 영상들 합치기 + 오디오 입히기
            if scene_sub_clips:
                try:
                    # 이미지 1, 2, 3을 순서대로 이어 붙임
                    full_scene_clip = concatenate_videoclips(scene_sub_clips, method="compose")
                    
                    # 오디오 설정 (길이가 미세하게 안 맞을 수 있으므로 오디오 길이에 맞춤)
                    full_scene_clip = full_scene_clip.set_audio(audio_clip)
                    
                    generated_clips.append(full_scene_clip)
                except Exception as e:
                    st.error(f"Scene {idx} 합치기 실패: {e}")
            
            progress_bar.progress((i + 1) / len(final_scenes))

        # Phase 3: Final Rendering (BGM Mixing 추가)
        if generated_clips:
            status_box.write("🎬 Phase 3: 영상 합치기 및 BGM 믹싱 중...")
            try:
                # 1. 컷 편집된 영상 연결
                final_video = concatenate_videoclips(generated_clips, method="compose")
                
                # 2. BGM 처리 로직
                bgm_path = get_bgm_path(bgm_mood)
                
                if bgm_path:
                    try:
                        # BGM 로드
                        bgm_clip = AudioFileClip(bgm_path)
                        
                        # 영상 길이에 맞춰 BGM 반복(Loop) 또는 자르기
                        # (영상보다 BGM이 짧으면 반복, 길면 자름)
                        if bgm_clip.duration < final_video.duration:
                            # 짝수 번 반복해서 충분히 길게 만듦
                            loop_count = int(final_video.duration // bgm_clip.duration) + 2
                            bgm_clip = concatenate_audioclips([bgm_clip] * loop_count)
                        
                        bgm_clip = bgm_clip.set_duration(final_video.duration)
                        
                        # 볼륨 조절 (가장 중요!)
                        # 목소리(Voice)는 100%, BGM은 10%~15% 수준으로 낮춤
                        voice_clip = final_video.audio.volumex(1.0) # 원본 목소리 유지
                        bgm_clip = bgm_clip.volumex(0.15)           # 배경음악 15% (은은하게)
                        
                        # 페이드 아웃 (끝날 때 2초간 서서히 작아짐)
                        bgm_clip = bgm_clip.audio_fadeout(2)
                        
                        # 오디오 합치기 (목소리 + BGM)
                        final_audio = CompositeAudioClip([voice_clip, bgm_clip])
                        final_video = final_video.set_audio(final_audio)
                        
                    except Exception as e:
                        st.warning(f"BGM 합성 중 오류 발생(영상은 BGM 없이 생성됩니다): {e}")

                # 3. 최종 내보내기
                safe_title = "".join([c for c in new_title if c.isalnum()]).strip() or "output"
                output_path = os.path.join(tempfile.gettempdir(), f"{safe_title}_final.mp4")
                
                final_video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', preset='ultrafast')
                
                status_box.update(label="✅ 영상 완성!", state="complete", expanded=False)
                st.balloons()
                st.success(f"🎉 '{new_title}' 영상이 완성되었습니다! (BGM: {bgm_mood})")
                st.video(output_path)
                
            except Exception as e:
                st.error(f"렌더링 오류: {e}")