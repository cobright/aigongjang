import streamlit as st
import os
import json
import requests
import fal_client
import google.generativeai as genai
from google.cloud import texttospeech
# 로컬 테스트용 dotenv (클라우드에서는 secrets 사용)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from moviepy.editor import *
import tempfile

# --- 1. 환경 및 UI 설정 ---
st.set_page_config(page_title="AI 영상 공장 (통합본)", page_icon="🎬", layout="wide")
st.title("🎬 AI 유튜브 영상 자동 생성기 (올인원 버전)")
st.markdown("주제만 입력하면 **대본(Gemini) + 이미지(Nano) + 음성(Google)**을 합쳐 영상을 만듭니다.")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정 패널")
    # API 키 상태 확인 (Secrets 또는 환경변수)
    api_status = {
        "Gemini API": "GOOGLE_API_KEY" in st.secrets or os.getenv("GOOGLE_API_KEY"),
        "Fal.ai API": "FAL_KEY" in st.secrets or os.getenv("FAL_KEY"),
        "Google TTS": "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    }
    for name, ready in api_status.items():
        if ready:
            st.success(f"{name}: 연결됨")
        else:
            st.error(f"{name}: 키가 없습니다!")
    
    st.divider()
    video_style = st.selectbox("화풍 선택", ["2D Cartoon", "Anime", "Cinematic Realistic", "Oil Painting"], index=0)
    num_scenes = st.slider("테스트할 씬 개수", min_value=1, max_value=5, value=2)
    # 캐릭터 일관성 (추후 구현)
    # ref_image = st.file_uploader("주인공 캐릭터 이미지 (일관성용)", type=['png', 'jpg'])

# --- 2. 내부 함수 정의 (모듈 통합) ---

def get_api_key(key_name):
    """Streamlit Secrets 또는 환경변수에서 API 키를 가져옵니다."""
    if key_name in st.secrets:
        return st.secrets[key_name]
    return os.getenv(key_name)

def generate_script_json(topic, num_scenes=3):
    """Gemini를 사용하여 구조화된 대본을 생성합니다."""
    api_key = get_api_key("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Gemini API 키가 없습니다.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        # 속도가 빠르고 안정적인 Flash 모델 사용 권장
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        YouTube Short Script for topic: '{topic}'.
        Output ONLY valid JSON. No Markdown. No ```json tags.
        Structure:
        {{
          "video_title": "Title",
          "scenes": [
            {{ "seq": 1, "narrative": "Voiceover text (Korean)", "visual_prompt": "Image prompt in English, {video_style} style" }},
            ... (Total {num_scenes} scenes)
          ]
        }}
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        # 실수로 마크다운이 붙어있을 경우 제거
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
            
        return json.loads(text)
    except Exception as e:
        st.error(f"🚨 Gemini 오류: {e}")
        return None

def generate_image(prompt, filename):
    """Nano Banana (Fal.ai)를 사용하여 이미지를 생성합니다."""
    # 클라우드 환경에서는 임시 폴더 사용
    output_path = os.path.join(tempfile.gettempdir(), filename)
    
    # Fal.ai 키 설정 (환경변수로 인식됨)
    os.environ["FAL_KEY"] = get_api_key("FAL_KEY")
    
    try:
        handler = fal_client.submit(
            "fal-ai/flux/dev",
            arguments={
                "prompt": prompt,
                "image_size": "landscape_16_9",
                "seed": 42, # 일관성을 위한 시드 고정
                "num_inference_steps": 25
            }
        )
        image_url = handler.get()['images'][0]['url']
        response = requests.get(image_url)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path
    except Exception as e:
        st.error(f"🚨 이미지 생성 오류 ({filename}): {e}")
        return None

def generate_audio(text, filename):
    """Google TTS를 사용하여 음성 파일을 생성합니다."""
    output_path = os.path.join(tempfile.gettempdir(), filename)
    
    try:
        # Streamlit Secrets에서 서비스 계정 JSON 정보를 가져옴
        creds_json = get_api_key("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not creds_json:
            st.error("❌ Google TTS JSON 키가 Secrets에 없습니다.")
            return None
            
        # JSON 문자열을 파싱하여 인증 정보로 사용
        from google.oauth2 import service_account
        creds_info = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
        client = texttospeech.TextToSpeechClient(credentials=credentials)

        input_text = texttospeech.SynthesisInput(text=text)
        # 한국어 남성 표준 음성 설정
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Standard-C",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        return output_path
    except Exception as e:
        st.error(f"🚨 TTS 오류 ({filename}): {e}")
        return None

# --- 3. 메인 실행 로직 ---

# 주제 입력
topic = st.text_input("영상 주제를 입력하세요", placeholder="예: 라면 맛있게 끓이는 법")

# 실행 버튼
if st.button("🚀 영상 생성 시작", type="primary"):
    if not topic:
        st.warning("주제를 입력해주세요!")
        st.stop()
    
    # 필수 API 키 확인
    if not (get_api_key("GOOGLE_API_KEY") and get_api_key("FAL_KEY")):
        st.error("필수 API 키(Gemini, Fal.ai)가 설정되지 않았습니다.")
        st.stop()

    result_container = st.container()

    # --- STEP 1: 기획 ---
    with st.status("🧠 1단계: Gemini가 대본을 기획 중입니다...", expanded=True) as status:
        script_data = generate_script_json(topic, num_scenes=num_scenes)
        
        if script_data:
            st.write("✅ 대본 생성 완료!")
            # 대본 내용 확인용 확장기
            with st.expander("📜 생성된 대본 보기"):
                st.json(script_data)
            video_title = script_data.get("video_title", "output")
            scenes = script_data.get("scenes", [])
            status.update(label="🧠 1단계 완료: 기획안 확정", state="complete", expanded=False)
        else:
            status.update(label="❌ 1단계 실패: 대본 생성 오류", state="error")
            st.stop()

    # --- STEP 2: 자산 생성 ---
    st.subheader("🎨 2단계: 자산 생성 (실시간 모니터링)")
    
    generated_clips = []
    progress_bar = st.progress(0)
    
    for i, scene in enumerate(scenes):
        seq = scene["seq"]
        narrative = scene["narrative"]
        visual_prompt = scene["visual_prompt"]
        
        # 진행률 업데이트
        progress_bar.progress((i) / len(scenes))
        
        with st.container():
            st.markdown(f"**🎬 Scene {seq} 작업 중...**")
            cols = st.columns([3, 2]) # 이미지와 오디오 영역 분할
            
            # 파일명 설정 (임시 폴더 경로 제외)
            img_name = f"temp_{seq:03d}.png"
            audio_name = f"temp_{seq:03d}.mp3"
            
            # A. 이미지 생성
            with cols[0]:
                with st.spinner("이미지 그리는 중..."):
                    img_path = generate_image(visual_prompt, img_name)
                if img_path:
                    st.image(img_path, caption=f"Scene {seq}")
            
            # B. 오디오 생성
            with cols[1]:
                with st.spinner("목소리 만드는 중..."):
                    audio_path = generate_audio(narrative, audio_name)
                if audio_path:
                    st.audio(audio_path)
            
            # C. 클립 데이터 저장
            if img_path:
                duration = 3 # 기본 길이 3초
                audioclip = None
                if audio_path:
                    try:
                        audioclip = AudioFileClip(audio_path)
                        duration = audioclip.duration
                    except Exception as e:
                         st.warning(f"오디오 클립 로드 실패 (기본 3초 적용): {e}")

                try:
                    # 이미지 클립 생성 및 길이 맞추기
                    img_clip = ImageClip(img_path).set_duration(duration)
                    if audioclip:
                        img_clip = img_clip.set_audio(audioclip)
                        # 팁: 이미지에 줌인 효과를 주면 더 역동적입니다. (선택 사항)
                        # img_clip = img_clip.resize(lambda t: 1 + 0.02*t) 
                    generated_clips.append(img_clip)
                except Exception as e:
                    st.error(f"클립 생성 중 오류: {e}")

    progress_bar.progress(100)

    # --- STEP 3: 편집 및 렌더링 ---
    if generated_clips:
        with st.spinner("🎞️ 3단계: 영상을 편집하고 렌더링 중입니다... (잠시만 기다리세요)"):
            try:
                # 영상 합치기
                final_video = concatenate_videoclips(generated_clips, method="compose")
                
                # 임시 출력 파일 경로
                output_file = os.path.join(tempfile.gettempdir(), f"output_{seq}.mp4")
                
                # 파일 쓰기 (fps 24, 코덱 설정)
                # preset='ultrafast'로 설정하여 렌더링 속도 향상
                final_video.write_videofile(output_file, fps=24, codec='libx264', audio_codec='aac', preset='ultrafast', logger=None)
                
                st.success("🎉 영상 제작이 완료되었습니다!")
                st.balloons()
                
                # --- 최종 결과물 출력 ---
                st.divider()
                st.header(f"📺 완성된 영상: {video_title}")
                st.video(output_file)
                
            except Exception as e:
                st.error(f"🚨 렌더링 오류 발생: {e}")
    else:
        st.error("❌ 생성된 클립이 없어 영상을 만들 수 없습니다.")