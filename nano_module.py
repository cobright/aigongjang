# nano_module.py
import os
import fal_client
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_image(prompt, filename, output_dir="assets/images"):
    """
    프롬프트를 받아 이미지를 생성하고 지정된 경로에 저장하는 함수
    """
    # 저장 경로 생성
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # 이미 존재하면 스킵 (PoC용)
    if os.path.exists(filepath):
        print(f"⏭️ 이미지 스킵: {filename} (이미 존재함)")
        return filepath

    print(f"🎨 나노바나나: 이미지 생성 중... ({filename})")
    
    try:
        # 1. Fal.ai API 호출 (Flux Dev 모델 사용 예시)
        # *참고: 실제 모델 경로는 Fal.ai 사이트에서 확인 필요 (예: "fal-ai/flux/dev")
        handler = fal_client.submit(
            "fal-ai/flux/dev",  # 모델 ID (변경될 수 있음)
            arguments={
                "prompt": prompt,
                "image_size": "landscape_16_9", # 유튜브 비율
                "seed": 42, # ✨ 일관성을 위한 시드 고정!
                "num_inference_steps": 30, # 퀄리티 조절 (높을수록 고퀄/느림)
            }
        )
        
        # 2. 결과 대기 및 이미지 URL 획득
        result = handler.get()
        image_url = result['images'][0]['url']
        
        # 3. 이미지 다운로드 및 저장
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✅ 이미지 저장 완료: {filepath}")
            return filepath
        else:
            print(f"❌ 이미지 다운로드 실패: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 나노바나나 오류: {e}")
        return None