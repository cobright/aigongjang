# gemini_module.py
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 기존 generate_script_json 함수를 지우고 이걸로 붙여넣으세요
def generate_script_json(topic, num_scenes=3):
    try:
        # 키 확인용 (키 앞 4자리만 출력해봄)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ GOOGLE_API_KEY가 없습니다. Secrets 설정을 확인하세요.")
            return None
        
        # 모델 설정
        genai.configure(api_key=api_key)
        
        # ⚠️ 모델 이름 변경: 'gemini-1.5-flash'가 가장 빠르고 에러가 적습니다.
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        prompt = f"""
        YouTube Short Script for topic: '{topic}'.
        Output ONLY valid JSON. No Markdown. No ```json tags.
        Structure:
        {{
          "video_title": "Title",
          "scenes": [
            {{ "seq": 1, "narrative": "Voiceover text", "visual_prompt": "Image prompt in English" }},
            ... (Total {num_scenes} scenes)
          ]
        }}
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # 실수로 마크다운(```json)이 붙어있으면 제거
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        return json.loads(text)
        
    except Exception as e:
        # ⭐ 여기가 핵심: 에러 내용을 화면에 그대로 보여줍니다.
        st.error(f"🚨 제미나이 에러 상세 내용: {e}")
        return None

# 테스트 실행
if __name__ == "__main__":
    sample_script = generate_script_json("인공지능이 바꿀 미래 직업")
    print(json.dumps(sample_script, indent=2, ensure_ascii=False))