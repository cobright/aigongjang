# gemini_module.py
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_script_json(topic, num_scenes=3):
    """
    주제를 받아 구조화된 JSON 대본을 생성하는 함수
    """
    # 1. 모델 설정 (Gemini 1.5 Pro 권장)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # 2. 시스템 프롬프트 설계 (JSON 출력 강제 및 구조 정의)
    system_prompt = f"""
    당신은 유튜브 영상 기획 전문가입니다. 주제를 바탕으로 영상 대본과 이미지 프롬프트를 작성해주세요.
    반드시 아래와 같은 엄격한 JSON 형식으로만 출력해야 합니다. 다른 설명은 생략하세요.
    
    [JSON 구조 예시]
    {{
      "video_title": "영상 제목",
      "scenes": [
        {{
          "seq": 1,
          "narrative": "성우가 읽을 내레이션 대본 (한 두 문장)",
          "visual_prompt": "이미지 생성용 영문 프롬프트 (웹툰 스타일, 캐릭터 특징 포함)",
          "mood": "분위기 (예: 밝음, 심각함)"
        }},
        ... (총 {num_scenes}개 씬 반복)
      ]
    }}
    
    주제: {topic}
    씬 개수: {num_scenes}개
    화풍: 2D 웹툰 스타일, 주인공은 파란색 후드티를 입은 밝은 표정의 청년으로 고정.
    """

    # 3. 생성 요청 및 응답 처리
    try:
        response = model.generate_content(system_prompt)
        # 응답 텍스트에서 JSON 부분만 추출 (혹시 모를 앞뒤 텍스트 제거)
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
            
        script_data = json.loads(response_text)
        print("✅ 제미나이: 대본 JSON 생성 완료!")
        return script_data
        
    except Exception as e:
        print(f"❌ 제미나이 오류: {e}")
        return None

# 테스트 실행
if __name__ == "__main__":
    sample_script = generate_script_json("인공지능이 바꿀 미래 직업")
    print(json.dumps(sample_script, indent=2, ensure_ascii=False))