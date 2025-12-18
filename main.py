# main.py
import os
from gemini_module import generate_script_json
from nano_module import generate_image
from tts_module import generate_audio
from moviepy.editor import *

def create_video_poc(topic):
    print(f"🚀 프로젝트 시작: 주제 - '{topic}'")
    
    # 1. 기획 단계 (Gemini)
    script_data = generate_script_json(topic, num_scenes=3) # PoC니까 3개만!
    if not script_data: return

    video_title = script_data.get("video_title", "output_video")
    scenes = script_data.get("scenes", [])
    
    clips = []
    
    # 2. 자산 생성 루프 (이미지 & 오디오)
    print("\n--- 🛠️ 자산 생성 시작 ---")
    for scene in scenes:
        seq = scene["seq"]
        narrative = scene["narrative"]
        visual_prompt = scene["visual_prompt"]
        
        # 파일명 정의 (001.png, 001.mp3 등)
        base_filename = f"{seq:03d}"
        image_filename = f"{base_filename}.png"
        audio_filename = f"{base_filename}.mp3"
        
        # a. 이미지 생성
        image_path = generate_image(visual_prompt, image_filename)
        # b. 오디오 생성
        audio_path = generate_audio(narrative, audio_filename)
        
        if image_path and audio_path:
            # 3. 클립 생성 (이미지 + 오디오 결합)
            # 이미지 클립을 만들고, 오디오 길이만큼 재생 시간을 설정합니다.
            audio_clip = AudioFileClip(audio_path)
            video_clip = ImageClip(image_path).set_duration(audio_clip.duration)
            video_clip = video_clip.set_audio(audio_clip)
            # *팁: 여기에 Ken Burns 효과(줌인/팬)를 추가하면 퀄리티가 확 올라갑니다!
            
            clips.append(video_clip)
        else:
            print(f"⚠️ 씬 {seq} 생성 실패. 건너뜁니다.")
            
    # 4. 최종 영상 조립 및 렌더링
    print("\n--- 🎬 최종 영상 렌더링 시작 ---")
    if clips:
        final_video = concatenate_videoclips(clips, method="compose")
        output_filename = f"{video_title.replace(' ', '_')}_poc.mp4"
        final_video.write_videofile(output_filename, fps=24)
        print(f"\n🎉✨ 축하합니다! 영상 제작이 완료되었습니다: {output_filename}")
    else:
        print("❌ 생성된 클립이 없습니다. 영상을 만들 수 없습니다.")

# 실행!
if __name__ == "__main__":
    # 원하는 주제를 입력하고 실행해보세요!
    topic_input = input("영상 주제를 입력하세요 (예: 라면 맛있게 끓이는 법): ")
    create_video_poc(topic_input)