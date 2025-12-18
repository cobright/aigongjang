import streamlit as st
import os
import json
from dotenv import load_dotenv
from moviepy.editor import *

# 우리가 만든 모듈들 가져오기
from gemini_module import generate_script_json
from nano_module import generate_image
from tts_module import generate_audio

# 1. 페이지 설정
st.set_page_config(page_title="AI 영상 공장", page_icon="🎬", layout="wide")
load_dotenv()

# 2. 사이드바: 설정 및 상태 확인
with st.sidebar:
    st.header("⚙️ 설정 패널")
    
    # API 키 상태 점검
    st.subheader("🔑 API 키 상태")
    if os.getenv("GOOGLE_API_KEY"):
        st.success("Gemini API: 연결됨")
    else:
        st.error("Gemini API 키가 없습니다!")

    if os.getenv("FAL_KEY"):
        st.success("Nano Banana (Fal.ai): 연결됨")
    else:
        st.error("Fal.ai 키가 없습니다!")
        
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        st.success("Google TTS: 연결됨")
    else:
        st.error("Google Cloud 키 파일이 없습니다!")

    st.divider()
    
    # 옵션 설정
    st.subheader("🎨 스타일 설정")
    video_style = st.selectbox("화풍 선택", ["2D Cartoon", "Anime", "Cinematic Realistic", "Oil Painting"])
    num_scenes = st.slider("테스트할 씬 개수", min_value=1, max_value=5, value=2)
    
    # 캐릭터 일관성 (나중에 구현할 기능을 위해 UI만 배치)
    ref_image = st.file_uploader("주인공 캐릭터 이미지 (일관성용)", type=['png', 'jpg'])


# 3. 메인 화면
st.title("🎬 AI 유튜브 영상 자동 생성기")
st.markdown("주제만 입력하면 **대본(Gemini) + 이미지(Nano) + 음성(Google)**을 합쳐 영상을 만듭니다.")

# 주제 입력
topic = st.text_input("영상 주제를 입력하세요", placeholder="예: 30대 직장인이 달러를 사야 하는 이유")

# 실행 버튼
if st.button("🚀 영상 생성 시작", type="primary"):
    if not topic:
        st.warning("주제를 입력해주세요!")
    else:
        # 작업 공간 초기화
        st.session_state['logs'] = []
        result_container = st.container()

        # --- STEP 1: 기획 ---
        with st.status("🧠 1단계: Gemini가 대본을 기획 중입니다...", expanded=True) as status:
            script_data = generate_script_json(topic, num_scenes=num_scenes)
            
            if script_data:
                st.write("✅ 대본 생성 완료!")
                st.json(script_data) # 대본 내용을 화면에 보여줌
                video_title = script_data.get("video_title", "output")
                scenes = script_data.get("scenes", [])
            else:
                st.error("대본 생성 실패. 로그를 확인하세요.")
                st.stop()
            
            status.update(label="🧠 1단계 완료: 기획안 확정", state="complete", expanded=False)

        # --- STEP 2: 자산 생성 ---
        st.subheader("🎨 2단계: 자산 생성 (실시간 모니터링)")
        
        generated_clips = []
        assets_col1, assets_col2 = st.columns(2) # 화면 분할
        
        progress_bar = st.progress(0)
        
        for i, scene in enumerate(scenes):
            seq = scene["seq"]
            # 프롬프트에 스타일 추가
            final_prompt = f"{scene['visual_prompt']}, {video_style} style"
            
            # 진행률 업데이트
            progress_bar.progress((i) / len(scenes))
            
            with st.container():
                st.markdown(f"**🎬 Scene {seq} 작업 중...**")
                
                # 파일명 설정
                img_name = f"temp_{seq:03d}.png"
                audio_name = f"temp_{seq:03d}.mp3"
                
                # A. 이미지 생성 (Nano Banana)
                img_path = generate_image(final_prompt, img_name)
                if img_path:
                    with assets_col1:
                        st.image(img_path, caption=f"Scene {seq} Image")
                
                # B. 오디오 생성 (Google TTS)
                audio_path = generate_audio(scene["narrative"], audio_name)
                if audio_path:
                    with assets_col2:
                        st.audio(audio_path)
                
                # 클립 데이터 저장
                if img_path and audio_path:
                    try:
                        audio_clip = AudioFileClip(audio_path)
                        # 이미지 클립 생성 및 길이 맞추기
                        img_clip = ImageClip(img_path).set_duration(audio_clip.duration)
                        img_clip = img_clip.set_audio(audio_clip)
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
                    output_file = f"output_{video_title.replace(' ', '_')}.mp4"
                    
                    # 파일 쓰기 (fps는 24로 설정)
                    final_video.write_videofile(output_file, fps=24, codec='libx264', audio_codec='aac')
                    
                    st.success("🎉 영상 제작이 완료되었습니다!")
                    st.balloons() # 풍선 효과 🎈
                    
                    # --- 최종 결과물 출력 ---
                    st.divider()
                    st.header(f"📺 완성된 영상: {video_title}")
                    st.video(output_file)
                    
                    # 다운로드 버튼
                    with open(output_file, 'rb') as f:
                        st.download_button('📥 영상 다운로드', f, file_name=output_file)
                        
                except Exception as e:
                    st.error(f"렌더링 오류 발생: {e}")
        else:
            st.error("생성된 클립이 없어 영상을 만들 수 없습니다.")