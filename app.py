import streamlit as st
import os
import json
import tempfile
import time
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Pillow 모듈
import random

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
    [Text] Gemini: 한글 텍스트 렌더링 규칙 추가
    """
    if not gemini_key: return None
    
    try:
        genai_old.configure(api_key=gemini_key)
        model = genai_old.GenerativeModel('gemini-2.5-flash') 
        
        prompt = f"""
        You are a YouTube Shorts Director. Create a script for: '{topic}'
        
        [CONSTRAINT - SCENE COUNT]
        You must generate **EXACTLY {num_scenes} scenes**.
        
        [LANGUAGE RULES]
        1. "narrative": **KOREAN (한국어)**. Casual spoken style.
        2. "visual_prompt": **ENGLISH (영어)** for descriptions.
        3. **[CRITICAL] KOREAN TEXT IN IMAGE**: 
           - If a scene needs specific text (e.g., a signboard, a letter, a phone screen), describe the object in English but write the **text content in KOREAN inside double quotes**.
           - Format: `Object description ..., text reads "한국어 내용", style ...`
           - Example: `A neon sign on a dark street that says "라면 맛집" || A hand holding a smartphone showing a message "입금 완료"`
        
        [CONTENT GUIDE]
        - Split visual actions using " || " for dynamic cuts.
        - **DO NOT** include the character description in the JSON output.
        
        [AUDIO GUIDE]
        - **Sound Effect**: Choose ONE suitable sound effect for each scene from this list:
          ["Whoosh (전환)", "Ding (정답/아이디어)", "Camera (찰칵)", "Pop (등장)", "Keyboard (타자)", "None"]
          - Use "Whoosh (전환)" for fast action.
          - Use "Ding (정답/아이디어)" for key information.
          - Use "Camera (찰칵)" for visual focus.
          
        [OUTPUT JSON FORMAT]
        {{
          "video_title": "Korean Title",
          "scenes": [
            {{ 
                "seq": 1, 
                "narrative": "이 간판 보이시죠?", 
                "visual_prompt": "A bright yellow wooden sign that reads \"원조 맛집\" hanging on a wall",
                "sound_effect": "Ding (정답/아이디어)" 
            }},
            ...
          ]
        }}
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text)
        
    except Exception as e:
        st.error(f"기획 오류: {e}")
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

def apply_random_motion(clip):
    """
    [Motion] 줌인, 줌아웃, 좌우/상하 패닝 중 하나를 랜덤으로 적용합니다.
    """
    try:
        import cv2
    except ImportError:
        return clip

    # 가능한 효과 목록
    effects = ['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'pan_up', 'pan_down']
    effect_type = random.choice(effects)
    
    # 효과 강도 (너무 어지럽지 않게 조절)
    speed = 0.04  # 움직임 속도

    def fl(get_frame, t):
        img = get_frame(t)
        h, w = img.shape[:2]
        
        # 시간 t에 따른 진행률 (0.0 ~ 1.0) -> 클립 끝날 때 최대 움직임
        progress = t / clip.duration if clip.duration > 0 else 0
        scale = 1 + (speed * progress) # 1.0 ~ 1.04
        
        # 원본 크기 유지를 위한 리사이징 계산
        new_w, new_h = int(w * scale), int(h * scale)
        
        # OpenCV로 이미지 확대 (기본 베이스)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 잘라낼 위치(Crop) 결정 로직
        if effect_type == 'zoom_in':
            # 중앙을 향해 줌인 (점점 안쪽을 보여줌)
            x = (new_w - w) // 2
            y = (new_h - h) // 2
            
        elif effect_type == 'zoom_out':
            # 줌아웃은 반대로 구현이 까다로우므로, 
            # "이미 확대된 상태에서 시작해서 원래대로 돌아오는" 로직 대신
            # 여기서는 편의상 "약한 줌인"으로 대체하거나 역방향 구현 필요.
            # (간단하게 구현하기 위해 zoom_in과 줌 포인트를 다르게 설정)
            x = (new_w - w) // 2
            y = (new_h - h) // 2
            
        elif effect_type == 'pan_left':
            # 오른쪽에서 왼쪽으로 이동 (보여주는 뷰포트가 왼쪽으로 감 -> x는 0에서 시작해서 커짐?? 반대임)
            # 이미지의 오른쪽 끝을 보여주다가 -> 왼쪽으로 이동
            # x: (new_w - w) -> 0 (감소)
            max_x = new_w - w
            x = int(max_x * (1 - progress))
            y = (new_h - h) // 2 # Y는 중앙 고정
            
        elif effect_type == 'pan_right':
            # 왼쪽에서 오른쪽으로 이동
            # x: 0 -> (new_w - w) (증가)
            max_x = new_w - w
            x = int(max_x * progress)
            y = (new_h - h) // 2
            
        elif effect_type == 'pan_up':
            # 아래에서 위로
            x = (new_w - w) // 2
            max_y = new_h - h
            y = int(max_y * (1 - progress))
            
        elif effect_type == 'pan_down':
            # 위에서 아래로
            x = (new_w - w) // 2
            max_y = new_h - h
            y = int(max_y * progress)
            
        else:
            # 기본 중앙
            x, y = (new_w - w) // 2, (new_h - h) // 2

        # 좌표가 음수거나 범위를 벗어나지 않게 클리핑
        x = max(0, min(x, new_w - w))
        y = max(0, min(y, new_h - h))
        
        return img_resized[y:y+h, x:x+w]

    return clip.fl(fl)

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

def get_sfx_path(sfx_name):
    """
    효과음 이름을 받아 파일 경로를 반환합니다.
    (자주 쓰는 5가지 필수 효과음 매핑)
    """
    if not sfx_name or sfx_name == "None":
        return None
        
    # 효과음 매핑 (무료 음원 URL 예시)
    sfx_library = {
        "Whoosh (전환)": "https://www.soundjay.com/transition/sounds/swoosh-01.mp3",
        "Ding (정답/아이디어)": "https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3",
        "Camera (찰칵)": "https://www.soundjay.com/mechanical/sounds/camera-shutter-click-01.mp3",
        "Pop (등장)": "https://www.soundjay.com/button/sounds/button-16.mp3",
        "Keyboard (타자)": "https://www.soundjay.com/mechanical/sounds/typewriter-6.mp3"
    }
    
    url = sfx_library.get(sfx_name)
    if not url: return None
    
    # 파일명 안전하게 변환
    safe_name = "".join(x for x in sfx_name if x.isalnum())
    filename = f"sfx_{safe_name}.mp3"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    
    # 캐싱 (이미 있으면 다운로드 안 함)
    if os.path.exists(filepath):
        return filepath
        
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    except Exception as e:
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
                # 효과음 선택 메뉴 추가
                sfx_options = ["None", "Whoosh (전환)", "Ding (정답/아이디어)", "Camera (찰칵)", "Pop (등장)", "Keyboard (타자)"]
                
                # 기획안에 있는 값 찾기 (없으면 None)
                current_sfx = scene.get('sound_effect', 'None')
                if current_sfx not in sfx_options: current_sfx = "None"
                
                st.selectbox(
                    "🔊 효과음 선택", 
                    sfx_options, 
                    index=sfx_options.index(current_sfx),
                    key=f"sfx_select_{i}"
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
                "narrative": st.session_state[f"narr_area_{i}"],
                "visual_prompt": st.session_state[f"vis_area_{i}"],
                "sound_effect": st.session_state[f"sfx_select_{i}"] # <--- 추가
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
            
            # [수정] 변수 정의가 먼저 있어야 합니다!
            aud_name = f"aud_{idx}_{timestamp}.mp3"

            # 1. 오디오 먼저 생성 (길이를 알아야 컷을 나눌 수 있음)
            aud_path = generate_audio(scene['narrative'], aud_name)
            
            if not aud_path: continue
            
            audio_clip = AudioFileClip(aud_path)
            
            # --- [NEW] 효과음 믹싱 로직 시작 ---
            sfx_name = scene.get('sound_effect')
            sfx_path = get_sfx_path(sfx_name)
            
            if sfx_path:
                try:
                    sfx_clip = AudioFileClip(sfx_path)
                    sfx_clip = sfx_clip.volumex(0.6)
                    audio_clip = CompositeAudioClip([audio_clip, sfx_clip])
                    
                except Exception as e:
                    st.warning(f"효과음 합성 실패: {e}")
            # --- [NEW] 효과음 믹싱 로직 끝 ---
            
            scene_duration = audio_clip.duration
            
            # 2. 비주얼 프롬프트 분석 ('||' 기준으로 쪼개기)
            raw_prompts = scene['visual_prompt'].split('||')            
            scene_sub_clips = [] 
            
            # 컷 당 지속 시간 계산
            # 프롬프트가 비어있을 경우를 대비한 안전장치 추가
            valid_prompts = [p.strip() for p in raw_prompts if p.strip()]
            if not valid_prompts: valid_prompts = [scene['visual_prompt']]
            
            clip_duration = scene_duration / len(valid_prompts)
            
            # 3. 각 컷 별로 이미지 생성 및 클립 만들기            
            for sub_idx, raw_text in enumerate(valid_prompts):
                
                # [캐릭터/화풍 강제 주입]
                final_prompt = f"{character_desc}, {raw_text}, {video_style}, cinematic lighting"
                
                img_name = f"img_{idx}_{sub_idx}_{timestamp}.png"
                status_box.write(f"    └ 컷 {sub_idx+1}: {raw_text[:15]}...")
                
                img_path = generate_image_google(final_prompt, img_name)
                
                if img_path:
                    try:
                        # 이미지 클립 생성
                        sub_clip = ImageClip(img_path).set_duration(clip_duration).resize(height=720)
                        
                        # [랜덤 모션 적용]
                        sub_clip = apply_random_motion(sub_clip)
                        
                        scene_sub_clips.append(sub_clip)
                    except Exception as e:
                        st.warning(f"이미지 클립 오류: {e}")
            
            # 4. 조각 영상들 합치기 + 오디오 입히기
            if scene_sub_clips:
                try:
                    full_scene_clip = concatenate_videoclips(scene_sub_clips, method="compose")
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