import streamlit as st
import os
import json
import tempfile
import time
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Pillow 모듈
import random
import textwrap

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

# --- [데이터 사전] 화풍 및 BGM 매핑 ---

# 1. 화풍 (Style) 매핑: 사용자가 선택하면 -> 전문 프롬프트로 변환
STYLE_PROMPTS = {
    "📸 실사: 시네마틱 (Cinematic)": "Cinematic shot, 4k, hyper-realistic, shallow depth of field, dramatic lighting, shot on Sony A7R",
    "📸 실사: 인스타 감성 (Aesthetic)": "Polaroid style, film grain, soft natural lighting, candid shot, aesthetic, VSCO filter",
    "🎨 2D: 웹툰/만화 (Webtoon)": "Korean webtoon style, cel shaded, vibrant colors, clean lines, anime style, manhwa",
    "🎨 3D: 픽사 스타일 (3D Animation)": "Disney Pixar style 3D render, cute, soft texture, volumetric lighting, Unreal Engine 5",
    "🖌️ 예술: 수채화 (Watercolor)": "Watercolor painting, soft brush strokes, pastel colors, artistic, dreamy",
    "🌃 사이버펑크 (Cyberpunk)": "Cyberpunk, neon lights, futuristic, dark atmosphere, glowing effects"
}

# --- [데이터 사전 추가] 장르별 최적화 설정 ---
GENRE_SETTINGS = {
    "📰 정보/뉴스 (Info)": {
        "persona": "Professional Journalist",
        "max_chars": 250, # 58초 꽉 채움
        "structure": "Hook (Shocking Fact) -> Body (3 Key Facts) -> Outro (Conclusion)",
        "tone": "Objective, clear, analytical, trustworthy",
        "pacing": "Fast and informative"
    },
    "👄 썰/스토리 (Story)": {
        "persona": "Friendly Storyteller",
        "max_chars": 210, # 연기할 시간 확보
        "structure": "Hook (Emotional Reaction) -> Body (Situation & Crisis) -> Outro (Twist/Ending)",
        "tone": "Casual, emotional, conversational (use '음슴체' or slang)",
        "pacing": "Dynamic with pauses for emphasis"
    },
    "🛍️ 리뷰/후기 (Review)": {
        "persona": "Sharp Product Reviewer",
        "max_chars": 180, # 제품 보여줄 시간 확보
        "structure": "Hook (Result first) -> Body (Pros & Cons) -> Outro (Final Rating)",
        "tone": "Honest, direct, trendy, critical",
        "pacing": "Moderate, focus on visuals"
    },
    "🕯️ 감성/동기부여 (Motivation)": {
        "persona": "Life Coach & Poet",
        "max_chars": 150, # 여백의 미
        "structure": "Hook (Deep Question) -> Body (Insight/Advice) -> Outro (Call to Action)",
        "tone": "Soft, warm, inspiring, calm",
        "pacing": "Slow, leaving space for music"
    }
}

# 2. BGM 매핑: 사용자가 선택하면 -> 무료 음원 URL로 변환
# (실제 운영 시에는 저작권 확인된 S3 링크나 Pexels/Youtube Audio Library 파일 권장)
BGM_URLS = {
    "🔇 없음 (Mute)": None,
    "☕ Lo-fi / 칠합 (Study)": "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3",
    "🌞 어쿠스틱 / 브이로그 (Daily)": "https://cdn.pixabay.com/download/audio/2022/03/24/audio_c8c8a73467.mp3", # 임시 URL (실제론 다른 파일 추천)
    "🏢 코퍼레이트 / 뉴스 (Info)": "https://cdn.pixabay.com/download/audio/2022/03/10/audio_c3d0b26f58.mp3",
    "🎬 시네마틱 / 웅장함 (Epic)": "https://cdn.pixabay.com/download/audio/2022/03/15/audio_736862b691.mp3",
    "🤪 펑키 / 예능 (Fun)": "https://cdn.pixabay.com/download/audio/2022/03/24/audio_823e8396d6.mp3"
}


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
    st.header("⚙️ 스튜디오 설정")
    
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
        
    # [NEW] Pexels 키 입력 추가
    # secrets.toml에 PEXELS_API_KEY가 있으면 그걸 쓰고, 없으면 입력창을 띄움
    pexels_key_env = get_secret("PEXELS_API_KEY")
    if not pexels_key_env:
        os.environ["PEXELS_API_KEY"] = st.text_input("Pexels API Key (스톡 영상용)", type="password")
    else:
        # 이미 환경변수에 있으면 성공 표시
        st.success("✅ Pexels API: Connected")
       
    st.divider() 
    # (사이드바 맨 위쪽, 주제 입력하는 곳 근처 혹은 설정 시작 부분)
    st.header("🎬 기획 설정")
    
    # [NEW] 장르 선택 메뉴
    selected_genre = st.selectbox(
        "영상 장르 (Genre)", 
        list(GENRE_SETTINGS.keys()), 
        index=0
    )
    
    # (선택된 장르에 대한 설명 표시 - 팁)
    st.info(f"💡 특징: {GENRE_SETTINGS[selected_genre]['tone']}")
    
    st.divider()    
    
    # [1] 주인공 페르소나 (4단 조립)
    st.subheader("👤 주인공 (Persona)")
    with st.expander("캐릭터 상세 설정 열기", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            char_age_gender = st.text_input("나이/성별/인종", value="20대 한국인 남성")
            char_outfit = st.text_input("의상 스타일", value="네이비 정장, 파란 넥타이")
        with col_b:
            char_hair = st.text_input("헤어/얼굴 특징", value="짧은 검은 머리, 안경")
            char_signature = st.text_input("시그니처 아이템", value="스마트워치")
            
        # 조립된 캐릭터 묘사 (이 변수가 AI에게 전달됨)
        character_desc = f"{char_age_gender}, {char_hair}, wearing {char_outfit}. Distinctive feature: {char_signature}"
        st.caption(f"📝 조합 결과: {character_desc}")

    # [2] 화풍 (Dictionary 활용)
    st.subheader("🎨 화풍 (Style)")
    selected_style_key = st.selectbox("스타일 선택", list(STYLE_PROMPTS.keys()), index=2)
    video_style = STYLE_PROMPTS[selected_style_key] # 실제 프롬프트로 변환
    
    # [3] 성우 (기존 유지)
    st.subheader("🎙️ 성우 (Voice)")
    voice_options = {
        "👨‍💼 남성 (차분한)": "ko-KR-Standard-C",
        "👩‍💼 여성 (차분한)": "ko-KR-Standard-A",
        "👧 여성 (발랄한)": "ko-KR-Standard-B",
        "👨 남성 (중저음)": "ko-KR-Standard-D"
    }
    selected_voice_label = st.selectbox("내레이터", list(voice_options.keys()), index=0)
    selected_voice_name = voice_options[selected_voice_label]

    # [4] BGM (Dictionary 활용)
    st.subheader("🎵 배경음악 (BGM)")
    bgm_mood = st.selectbox("분위기 선택", list(BGM_URLS.keys()), index=1)
    
    # [NEW] 자막 ON/OFF 기능 추가
    st.subheader("📝 자막 (Subtitles)")
    use_subtitles = st.checkbox("자막 포함 (Subtitles)", value=True) # 기본값은 켜짐
    
    st.divider()
    num_scenes = st.slider("씬(Scene) 개수", 2, 8, 4)
    

# --- 2. 핵심 모듈 함수 ---
# [수정] genre_key 인자 추가
def generate_script_json(topic, num_scenes, genre_key):
    """
    [Planning] 장르별 최적화 로직이 적용된 기획안 생성
    """
    if not gemini_key: return None
    
    # 선택된 장르의 설정 가져오기
    settings = GENRE_SETTINGS.get(genre_key, GENRE_SETTINGS["📰 정보/뉴스 (Info)"])
    
    try:
        genai_old.configure(api_key=gemini_key)
        model = genai_old.GenerativeModel('gemini-2.5-flash') 
        
        prompt = f"""
        You are a {settings['persona']} specialized in creating viral YouTube Shorts.
        Create a script for the topic: '{topic}'
        
        [GENRE SPECIFIC RULES]
        - **Genre**: {genre_key}
        - **Tone**: {settings['tone']}
        - **Structure Strategy**: Follow {settings['structure']}
        - **Length Constraint**: Keep the Korean narrative STRICTLY under **{settings['max_chars']} characters** (including spaces). This is critical for video pacing.
        
        [CONSTRAINT - SCENE COUNT]
        Generate exactly {num_scenes} scenes.
        
        [LANGUAGE RULES]
        1. "narrative": **KOREAN (한국어)**. Style must match the Tone ({settings['tone']}).
        2. "visual_prompt": **KOREAN (한국어)**.
        3. **Visual Strategy**:
           - If the genre is 'Review' or 'Info', focus on showing the object/fact clearly.
           - If the genre is 'Story' or 'Motivation', focus on facial expressions and atmosphere.
           - Use `[VIDEO] keyword` for generic scenes (Sky, City, Coffee).
        
        [OUTPUT JSON FORMAT]
        {{
          "video_title": "Title in Korean",
          "scenes": [
            {{ "seq": 1, "narrative": "Korean script...", "visual_prompt": "Korean description..." }},
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

def generate_image_google(prompt, filename, ref_image_path=None):
    """
    [Image] Gemini 3 Pro Image: 레퍼런스 이미지를 활용한 캐릭터 고정
    """
    if not gemini_key: return None
    
    output_path = os.path.join(tempfile.gettempdir(), filename)

    try:
        client = genai.Client(api_key=gemini_key)
        # 검색 결과에 따른 최신 모델명 (2025년 기준)
        model_id = "gemini-3-pro-image-preview"

        # 1. 프롬프트 구성 (텍스트)
        # 캐릭터 일관성을 위해 'Consistent character' 키워드 강조
        contents_parts = [types.Part.from_text(text=prompt + ", consistent character identity, high fidelity")]

        # 2. 레퍼런스 이미지 추가 (핵심 기능)
        # 만약 참조 이미지가 있다면, 프롬프트와 함께 AI에게 보여줍니다.
        if ref_image_path and os.path.exists(ref_image_path):
            with open(ref_image_path, "rb") as f:
                img_data = f.read()
                # 텍스트 + 이미지를 멀티모달 입력으로 전달
                contents_parts.append(types.Part.from_bytes(data=img_data, mime_type="image/png"))

        contents = [types.Content(role="user", parts=contents_parts)]
        
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(image_size="1K"), # 1024x1024
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
        
        return None

    except Exception as e:
        # Gemini 3 모델이 아직 배포되지 않은 리전일 경우 안전장치
        if "404" in str(e) or "not found" in str(e).lower():
            st.warning("⚠️ Gemini 3 모델을 찾을 수 없어 2.0 Flash로 대체합니다.")
            # (여기에 기존 generate_image_google 로직의 Fallback 코드를 넣거나 재귀 호출 가능)
        st.warning(f"이미지 생성 오류: {e}")
        return None

def generate_audio(text, filename, voice_name="ko-KR-Standard-C"):
    """
    [Voice] Google TTS: 성우 선택 기능 추가
    """
    output_path = os.path.join(tempfile.gettempdir(), filename)
    
    # 인증 (기존 로직 유지)
    credentials = None
    if tts_key_json:
        try:
            creds_info = json.loads(tts_key_json, strict=False)
            credentials = service_account.Credentials.from_service_account_info(creds_info)
        except: return None
    elif tts_key_path and os.path.exists(tts_key_path):
        credentials = service_account.Credentials.from_service_account_file(tts_key_path)
    else:
        return None

    try:
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        input_text = texttospeech.SynthesisInput(text=text)
        
        # [핵심 수정] 전달받은 voice_name 적용
        # 성별(Gender)은 목소리 이름에 맞춰 자동 설정
        if "Standard-A" in voice_name or "Standard-B" in voice_name:
            gender = texttospeech.SsmlVoiceGender.FEMALE
        else:
            gender = texttospeech.SsmlVoiceGender.MALE
            
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR", 
            name=voice_name, 
            ssml_gender=gender
        )
        
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

def get_bgm_path(mood_key):
    """
    선택된 BGM 키에 해당하는 URL을 다운로드합니다.
    """
    if not mood_key or mood_key == "🔇 없음 (Mute)":
        return None
    
    url = BGM_URLS.get(mood_key)
    if not url: return None
    
    # [핵심 수정 1] 파일명에서 한글 제거 (괄호 안의 영어 키워드만 추출)
    # 예: "🌞 어쿠스틱 / 브이로그 (Daily)" -> "Daily"
    try:
        if '(' in mood_key:
            english_key = mood_key.split('(')[-1].replace(')', '').strip()
        else:
            english_key = "default"
    except:
        english_key = "bgm"
        
    # 영문/숫자만 남기기
    safe_name = "".join(x for x in english_key if x.isalnum())
    filename = f"bgm_{safe_name}.mp3"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    
    # 캐싱 확인 및 유효성 검사
    if os.path.exists(filepath):
        # 파일이 있는데 크기가 너무 작으면(1KB 미만) 삭제하고 다시 받음
        if os.path.getsize(filepath) < 1000:
            os.remove(filepath)
        else:
            return filepath
        
    try:
        # [핵심 수정 2] 다운로드 검증
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ BGM 다운로드 실패(Status): {mood_key}")
            return None
            
        # 내용이 HTML 에러페이지인지 확인 (크기 체크)
        if len(response.content) < 1000:
            print(f"❌ BGM 파일 손상(Too small): {mood_key}")
            return None
            
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    except Exception as e:
        print(f"❌ BGM 네트워크 오류: {e}")
        return None

def get_sfx_path(sfx_name):
    """
    [안전 버전] 효과음 다운로드 및 검증
    """
    if not sfx_name or sfx_name == "None":
        return None
        
    # 1. 효과음 URL 매핑 (접근이 더 원활한 GitHub 소스 등으로 대체 권장)
    # 아래는 예시용 URL이며, 실제 서비스시 본인의 S3나 호스팅 URL을 넣는 것이 가장 안전합니다.
    sfx_library = {
        "Whoosh (전환)": "https://cdn.pixabay.com/download/audio/2022/03/24/audio_c8c8a73467.mp3", # Pixabay Free
        "Ding (정답/아이디어)": "https://cdn.pixabay.com/download/audio/2022/03/15/audio_736862b691.mp3",
        "Camera (찰칵)": "https://cdn.pixabay.com/download/audio/2022/03/10/audio_c3d0b26f58.mp3",
        "Pop (등장)": "https://cdn.pixabay.com/download/audio/2022/03/10/audio_c8c8a73467.mp3", # 임시 대체
        "Keyboard (타자)": "https://cdn.pixabay.com/download/audio/2022/03/24/audio_823e8396d6.mp3"
    }
    
    url = sfx_library.get(sfx_name)
    if not url: return None
    
    # 2. 파일명 안전하게 변환 (한글 제거, 순수 영문/숫자만 남김)
    # 예: "Pop (등장)" -> "Pop"
    safe_key = sfx_name.split('(')[0].strip() # 괄호 앞부분만 가져옴
    safe_name = "".join(x for x in safe_key if x.isalnum())
    filename = f"sfx_{safe_name}.mp3"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    
    # 3. 캐싱 및 다운로드 검증
    if not os.path.exists(filepath):
        try:
            # 타임아웃 3초 설정
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=3)
            
            # (중요) 요청이 성공했는지 확인
            if response.status_code != 200:
                print(f"❌ 효과음 다운로드 실패(HTTP Error): {sfx_name}")
                return None
                
            # (중요) 파일 내용이 너무 작으면(1KB 미만) 가짜 파일(HTML 에러페이지)일 확률 높음
            if len(response.content) < 1000:
                print(f"❌ 효과음 파일 손상 의심(Too small): {sfx_name}")
                return None
                
            with open(filepath, "wb") as f:
                f.write(response.content)
                
        except Exception as e:
            print(f"❌ 효과음 네트워크 오류: {e}")
            return None
            
    return filepath

# --- (get_sfx_path 함수 아래에 추가) ---

def get_korean_font():
    """
    한글 폰트(나눔고딕)를 다운로드하여 경로를 반환합니다.
    """
    font_path = os.path.join(tempfile.gettempdir(), "NanumGothic-Bold.ttf")
    
    # 이미 있으면 반환
    if os.path.exists(font_path):
        return font_path
        
    # 없으면 다운로드 (Google Fonts)
    url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf"
    try:
        response = requests.get(url, timeout=10)
        with open(font_path, "wb") as f:
            f.write(response.content)
        return font_path
    except Exception:
        return None

def create_subtitle_clip(text, duration, font_path):
    """
    [개선됨] 긴 문장 자동 줄바꿈 + 중앙 정렬 + 폰트 크기 최적화
    """
    try:
        w, h = 1280, 720
        # 투명 캔버스 생성
        img = PIL.Image.new('RGBA', (w, h), (255, 255, 255, 0))
        draw = PIL.ImageDraw.Draw(img)
        
        # 1. 폰트 설정 (크기를 55 -> 40으로 줄임)
        font_size = 40 
        try:
            if font_path:
                font = PIL.ImageFont.truetype(font_path, font_size)
            else:
                font = PIL.ImageFont.load_default()
        except:
            font = PIL.ImageFont.load_default()

        # 2. 자동 줄바꿈 (핵심!)
        # 화면 폭에 맞춰서 약 30~35글자마다 줄을 바꿈
        wrapped_text = textwrap.fill(text, width=35)

        # 3. 글자 크기 및 위치 계산 (여러 줄일 경우를 대비해 multiline 사용)
        # 텍스트 박스 크기 구하기
        left, top, right, bottom = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
        text_w = right - left
        text_h = bottom - top
        
        # 정중앙 하단 위치 계산
        x = (w - text_w) / 2
        y = h - text_h - 50 # 바닥에서 50px 위 (여유 공간 확보)

        # 4. 그리기 (테두리 포함)
        draw.multiline_text(
            (x, y), 
            wrapped_text, 
            font=font, 
            fill="white", 
            stroke_width=3, 
            stroke_fill="black", 
            align="center"
        )
        
        return ImageClip(np.array(img)).set_duration(duration)
        
    except Exception as e:
        print(f"자막 생성 오류: {e}")
        return None

# --- (create_subtitle_clip 함수 아래에 추가) ---

def get_pexels_video(query, duration):
    """
    [Stock Video] Pexels API를 사용하여 무료 영상을 다운로드하고,
    오디오 길이에 맞게 편집(Loop/Cut)하여 반환합니다.
    """
    # 1. API 키 확인 (사이드바 입력값 또는 Secrets)
    api_key = get_secret("PEXELS_API_KEY") 
    if not api_key:
        st.error("❌ Pexels API Key가 없습니다. 사이드바 설정을 확인하세요.")
        return None
        
    # 2. 검색 요청
    headers = {'Authorization': api_key}
    # landscape(가로), medium(중간화질) 설정으로 전송량 절약
    params = {'query': query, 'per_page': 1, 'orientation': 'landscape', 'size': 'medium'}
    
    try:
        response = requests.get('https://api.pexels.com/videos/search', headers=headers, params=params, timeout=10)
        data = response.json()
        
        if not data.get('videos'):
            return None # 검색 결과 없음
            
        # 3. 영상 URL 추출 (가장 적당한 화질 선택)
        video_files = data['videos'][0]['video_files']
        # 너비가 1280에 가까운 파일 찾기 (HD급)
        target_video = min(video_files, key=lambda x: abs(x['width'] - 1280))
        video_url = target_video['link']
        
        # 4. 다운로드 및 캐싱
        safe_name = "".join(x for x in query if x.isalnum())
        filename = f"pexels_{safe_name}.mp4"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        if not os.path.exists(filepath):
            vid_response = requests.get(video_url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in vid_response.iter_content(chunk_size=1024):
                    if chunk: f.write(chunk)
                    
        # 5. MoviePy 클립 변환 및 길이 맞춤
        clip = VideoFileClip(filepath)
        
        # 소리 제거 (TTS와 겹치므로)
        clip = clip.without_audio()
        
        # 길이 맞추기 로직
        if clip.duration < duration:
            # 영상이 짧으면 반복(Loop)
            # vfx.loop는 최신 버전에서 방식이 다를 수 있어 안전하게 수동 반복
            loop_count = int(duration // clip.duration) + 2
            clip = concatenate_videoclips([clip] * loop_count)
            
        # 필요한 길이만큼 자르기
        clip = clip.subclip(0, duration)
        
        # 720p로 리사이징
        clip = clip.resize(height=720)
        
        # 만약 비율이 안 맞으면 중앙 크롭 (16:9 강제)
        # (간단하게 구현하기 위해 resize만 적용, 필요시 crop 추가 가능)
        
        return clip

    except Exception as e:
        print(f"Pexels 다운로드 실패: {e}")
        return None

def generate_video_veo(prompt, filename):
    """
    [NEW] Google Veo (Text-to-Video)를 사용하여 영상을 생성합니다.
    """
    if not gemini_key: return None
    
    output_path = os.path.join(tempfile.gettempdir(), filename)
    
    # 이미 생성된 파일이 있으면 캐시 사용 (Veo는 비싸고 오래 걸리므로)
    if os.path.exists(output_path):
        return output_path

    try:
        client = genai.Client(api_key=gemini_key)
        
        # [중요] 2025년 12월 기준 최신 모델 ID (상황에 따라 'veo-3.1-generate-preview-1015' 등으로 변경 필요)
        model_id = "veo-3.1-generate-preview" 

        # 영상 생성 설정
        # 24fps, 1080p 등 설정 가능
        generate_config = types.GenerateContentConfig(
            response_modalities=["VIDEO"], # 응답 형식을 비디오로 설정
            video_config=types.VideoConfig(
                aspect_ratio="16:9", 
                sample_count=1, 
                seconds=6 # 씬당 6초 생성
            )
        )

        prompt_text = f"Cinematic movie shot, {prompt}, high quality, 4k"

        # 생성 요청 (시간이 30초~1분 정도 소요될 수 있음)
        response = client.models.generate_content(
            model=model_id,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])],
            config=generate_config
        )
        
        # 응답 처리
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data: # 비디오 데이터가 인라인으로 오는 경우
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    return output_path
                elif part.file_data: # 파일 URI로 오는 경우 (Vertex AI 등)
                    # 파일 다운로드 로직 필요 (SDK 버전에 따라 다름. 여기선 인라인 가정)
                    pass
        
        return None

    except Exception as e:
        print(f"Veo 생성 실패: {e}")
        # 실패 시 None을 반환하여 기존 AI 이미지(백업)로 넘어가게 함
        return None
    
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
        script_data = generate_script_json(topic, num_scenes, selected_genre)
        
        if script_data:
            st.session_state['script_data'] = script_data
            st.rerun()
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
        
        # --- Phase 2: Veo + Stock Video + AI Image 하이브리드 ---
        status_box.write("🎨 Phase 2: 캐릭터 기준 이미지(Anchor) 생성 중...")
        # [Step 0] 기준 캐릭터 이미지 생성 (이 이미지가 영상 내내 쓰임)
        # 가장 자세한 묘사 + 정면 얼굴 위주
        anchor_prompt = f"A detailed character sheet of {character_desc}, {video_style}, neutral expression, front view, white background"
        anchor_img_name = f"anchor_char_{int(time.time())}.png"
        
        # 첫 번째 생성 시에는 레퍼런스가 없으므로 None
        anchor_image_path = generate_image_google(anchor_prompt, anchor_img_name, ref_image_path=None)
        
        if anchor_image_path:
            st.image(anchor_image_path, caption="✅ 생성된 기준 캐릭터 (이 얼굴로 고정됩니다)", width=200)
        else:
            st.warning("기준 캐릭터 생성 실패. 일관성이 떨어질 수 있습니다.")

        progress_bar = st.progress(0)
        generated_clips = []
        
        korean_font_path = get_korean_font()
        
        for i, scene in enumerate(final_scenes):
            idx = scene['seq']
            status_box.write(f"  - Scene {idx} 작업 중...")
            
            timestamp = int(time.time())
            aud_name = f"aud_{idx}_{timestamp}.mp3"
            
            # 1. 오디오 생성
            aud_path = generate_audio(scene['narrative'], aud_name, voice_name=selected_voice_name)
            if not aud_path: continue
            audio_clip = AudioFileClip(aud_path)
            
            # 효과음 믹싱
            sfx_name = scene.get('sound_effect')
            sfx_path = get_sfx_path(sfx_name)
            if sfx_path and os.path.exists(sfx_path):
                try:
                    sfx_clip = AudioFileClip(sfx_path).volumex(0.6)
                    audio_clip = CompositeAudioClip([audio_clip, sfx_clip])
                except: pass
            
            scene_duration = audio_clip.duration
            visual_prompt = scene['visual_prompt'].strip()
            scene_final_clip = None

            # ==========================================
            # [전략 1] 스톡 비디오 (태그가 있는 경우 최우선)
            # ==========================================
            if visual_prompt.upper().startswith("[VIDEO]"):
                search_query = visual_prompt[7:].strip()
                status_box.write(f"    🎥 스톡 비디오 검색: {search_query}")
                scene_final_clip = get_pexels_video(search_query, scene_duration)
                
                if not scene_final_clip:
                    status_box.warning("스톡 비디오 실패 -> Veo 생성 시도")
                    visual_prompt = search_query # 태그 떼고 Veo로 넘김

            # ==========================================
            # [전략 2] Google Veo (진짜 생성형 비디오)
            # ==========================================
            if scene_final_clip is None:
                # 캐릭터 일관성을 위한 프롬프트 조합
                veo_prompt = f"{character_desc}, {visual_prompt}, {video_style}, consistent character"
                vid_name = f"veo_{idx}_{timestamp}.mp4"
                
                status_box.write(f"    🎬 Veo 영상 생성 중... (약 30초 소요)")
                veo_path = generate_video_veo(veo_prompt, vid_name)
                
                if veo_path:
                    try:
                        # Veo 영상 로드
                        veo_clip = VideoFileClip(veo_path)
                        # 소리가 있을 수 있으므로 제거 (TTS 사용 위해)
                        veo_clip = veo_clip.without_audio()
                        
                        # 길이 맞추기 (Loop or Cut)
                        if veo_clip.duration < scene_duration:
                             loop_count = int(scene_duration // veo_clip.duration) + 2
                             veo_clip = concatenate_videoclips([veo_clip] * loop_count)
                        
                        scene_final_clip = veo_clip.subclip(0, scene_duration).resize(height=720)
                        status_box.write("    ✅ Veo 생성 성공!")
                    except Exception as e:
                        st.warning(f"Veo 클립 처리 오류: {e}")

            # ==========================================
            # [전략 3] AI 이미지 (Veo 실패 시 백업)
            # ==========================================
            if scene_final_clip is None:
                status_box.write("    🎨 AI 이미지 모드 (백업) 실행")
                # (기존 이미지 컷 쪼개기 로직 유지)
                raw_prompts = visual_prompt.split('||')
                valid_prompts = [p.strip() for p in raw_prompts if p.strip()]
                if not valid_prompts: valid_prompts = [visual_prompt]
                
                clip_duration = scene_duration / len(valid_prompts)
                scene_sub_clips = []
                
                for sub_idx, raw_text in enumerate(valid_prompts):
                    final_prompt = f"{character_desc}, {raw_text}, {video_style}"
                    img_name = f"img_{idx}_{sub_idx}_{timestamp}.png"
                    
                    # [핵심] 여기서 anchor_image_path를 넘겨줍니다!
                    img_path = generate_image_google(final_prompt, img_name, ref_image_path=anchor_image_path)
                    
                    if img_path:
                        try:
                            sub_clip = ImageClip(img_path).set_duration(clip_duration).resize(height=720)
                            sub_clip = apply_random_motion(sub_clip)
                            scene_sub_clips.append(sub_clip)
                        except: pass
                
                if scene_sub_clips:
                    scene_final_clip = concatenate_videoclips(scene_sub_clips, method="compose")

            # 최종 합성 (오디오 + 자막 + 트랜지션)
            if scene_final_clip:
                try:
                    scene_final_clip = scene_final_clip.set_audio(audio_clip)
                    
                    if use_subtitles:
                        subtitle_clip = create_subtitle_clip(scene['narrative'], scene_final_clip.duration, korean_font_path)
                        if subtitle_clip:
                            scene_final_clip = CompositeVideoClip([scene_final_clip, subtitle_clip])
                    
                    scene_final_clip = scene_final_clip.fadein(0.5)
                    generated_clips.append(scene_final_clip)
                except Exception as e:
                    st.error(f"최종 합성 실패: {e}")
            
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