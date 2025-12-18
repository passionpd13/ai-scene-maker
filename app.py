import streamlit as st
import requests
import json
import time
import os
import re
import shutil
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from google import genai
from google.genai import types

# ==========================================
# [설정] 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="AI 영상 씬 생성기 (Pro)", layout="wide", page_icon="🎬")

# 파일 저장 경로 설정
BASE_PATH = "./web_result_files"
IMAGE_OUTPUT_DIR = os.path.join(BASE_PATH, "output_images")

# 텍스트 모델은 고정 (가장 성능 좋은 것)
GEMINI_TEXT_MODEL_NAME = "gemini-2.5-pro"

# ==========================================
# [함수] 로직 처리
# ==========================================

def init_folders():
    """결과 폴더 초기화"""
    if os.path.exists(IMAGE_OUTPUT_DIR):
        try:
            shutil.rmtree(IMAGE_OUTPUT_DIR)
        except Exception:
            pass
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

def split_script_by_time(script, chars_per_chunk=100):
    """대본 분할 로직"""
    temp_sentences = script.replace(".", ".|").replace("?", "?|").replace("!", "!|").split("|")
    chunks = []
    current_chunk = ""
    for sentence in temp_sentences:
        sentence = sentence.strip()
        if not sentence: continue
        if len(current_chunk) + len(sentence) < chars_per_chunk:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def make_filename(scene_num, text_chunk):
    """파일명 생성"""
    clean_line = text_chunk.replace("\n", " ").strip()
    clean_line = re.sub(r'[\\/:*?"<>|]', "", clean_line)
    words = clean_line.split()
    
    if len(words) <= 6:
        summary = " ".join(words)
    else:
        start_part = " ".join(words[:3])
        end_part = " ".join(words[-3:])
        summary = f"{start_part}...{end_part}"
    
    filename = f"S{scene_num:03d}_{summary}.png"
    return filename

def generate_prompt(api_key, index, text_chunk, style_instruction):
    """텍스트 프롬프트 생성"""
    scene_num = index + 1
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TEXT_MODEL_NAME}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}

    full_instruction = f"""
    [Role]
    You are an expert AI art director.

    [Style Guideline]
    {style_instruction}

    [Task]
    Create a detailed image generation prompt based on the provided script chunk.
    Describe the scene visually in English. Focus on the visual elements described in the Style Guideline.
    Output ONLY the prompt text.
    """
    
    payload = {
        "contents": [{"parts": [{"text": f"Instruction:\n{full_instruction}\n\nScript Segment:\n\"{text_chunk}\"\n\nImage Prompt:"}]}]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            try:
                prompt = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            except:
                prompt = text_chunk
            return (scene_num, prompt)
        elif response.status_code == 429:
            time.sleep(2)
            return (scene_num, f"Scene depicting: {text_chunk}")
        else:
            return (scene_num, f"Error generating prompt: {response.status_code}")
    except Exception as e:
        return (scene_num, f"Error: {e}")

def generate_image(client, prompt, filename, output_dir, selected_model_name):
    """이미지 생성 함수 (선택된 모델 사용)"""
    full_path = os.path.join(output_dir, filename)
    try:
        # 선택된 모델명(selected_model_name)을 사용하여 호출
        response = client.models.generate_content(
            model=selected_model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio="16:9")
            )
        )
        
        if response.parts:
            for part in response.parts:
                if part.inline_data:
                    img_data = part.inline_data.data
                    image = Image.open(BytesIO(img_data))
                    image.save(full_path)
                    return full_path
        return None

    except Exception as e:
        print(f"이미지 생성 실패 ({filename}): {e}")
        return None

def create_zip_buffer(source_dir):
    """ZIP 압축"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.basename(file_path))
    buffer.seek(0)
    return buffer

# ==========================================
# [UI] 사이드바 설정 영역
# ==========================================
with st.sidebar:
    st.header("⚙️ 환경 설정")
    
    api_key = st.text_input("🔑 Google API Key", type="password", help="Gemini API 키를 입력하세요.")
    
    st.markdown("---")

    # ★★★ [추가] 이미지 모델 선택 버튼 ★★★
    st.subheader("🖼️ 이미지 모델 선택")
    model_choice = st.radio(
        "사용할 AI 모델을 선택하세요:",
        ("Premium (Gemini 3 Pro)", "Fast (Gemini 2.5 Flash)"),
        index=0 # 기본값: 3 Pro
    )

    # 선택에 따라 모델명 변수 설정
    if "Gemini 3 Pro" in model_choice:
        SELECTED_IMAGE_MODEL = "gemini-3-pro-image-preview"
    else:
        SELECTED_IMAGE_MODEL = "gemini-2.5-flash-image"
        
    st.info(f"✅ 현재 선택된 모델:\n`{SELECTED_IMAGE_MODEL}`")

    st.markdown("---")

    st.subheader("⏱️ 장면 분할 설정")
    chunk_duration = st.slider("한 장면당 지속 시간 (초)", 10, 60, 20, 5)
    chars_limit = chunk_duration * 8 
    st.caption(f"약 **{chars_limit}글자** 단위로 분할됩니다.")

    st.markdown("---")
    
    st.subheader("🎨 스타일 지침")
    default_style = """
대사에 어울리는 2d 얼굴이 둥근 하얀색 스틱맨 연출로 설명과 이해가 잘되는 화면 자료 느낌으로 그려줘 상황을 잘 나타내게 분활화면으로 말고 하나의 장면으로
너무 어지럽지 않게, 글씨는 핵심 키워드 2~3만 나오게 한다
글씨가 너무 많지 않게 핵심만. 2D 스틱맨을 활용해 대본을 설명이 잘되게 설명하는 연출을 한다. 자막 스타일 연출은 하지 않는다.
글씨가 나올경우 핵심 키워드 중심으로만 나오게 너무 글이 많지 않도록 한다, 글자는 배경과 서물에 자연스럽게 연출, 전체 배경 연출은 2D로 디테일하게 몰입감 있게 연출해서 그려줘 (16:9)
    """
    style_instruction = st.text_area("AI에게 지시할 그림 스타일", value=default_style.strip(), height=200)

    st.markdown("---")
    max_workers = st.slider("작업 속도(병렬 수)", 1, 10, 5)

# ==========================================
# [UI] 메인 화면
# ==========================================
st.title("🎬 AI 대본 시각화 도구 (Pro)")
st.caption(f"🔧 Text: {GEMINI_TEXT_MODEL_NAME} | 🎨 Image: {SELECTED_IMAGE_MODEL}")

script_input = st.text_area("📜 대본을 입력하세요", height=200, placeholder="대본 붙여넣기...")

if 'generated_results' not in st.session_state:
    st.session_state['generated_results'] = []
if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False

start_btn = st.button("🚀 이미지 생성 시작", type="primary", use_container_width=True)

if start_btn:
    if not api_key:
        st.error("⚠️ API Key를 입력해주세요.")
    elif not script_input:
        st.warning("⚠️ 대본을 입력해주세요.")
    else:
        st.session_state['is_processing'] = True
        st.session_state['generated_results'] = [] 
        
        init_folders()
        client = genai.Client(api_key=api_key)
        
        status_box = st.status("작업 진행 중...", expanded=True)
        progress_bar = st.progress(0)
        
        # 1. 대본 분할
        status_box.write(f"✂️ 대본 분할 중...")
        chunks = split_script_by_time(script_input, chars_per_chunk=chars_limit)
        total_scenes = len(chunks)
        status_box.write(f"✅ {total_scenes}개 장면으로 분할 완료.")
        
        # 2. 프롬프트 생성
        status_box.write(f"📝 프롬프트 작성 중 ({GEMINI_TEXT_MODEL_NAME})...")
        prompts = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                futures.append(executor.submit(generate_prompt, api_key, i, chunk, style_instruction))
            
            for i, future in enumerate(as_completed(futures)):
                prompts.append(future.result())
                progress_bar.progress((i + 1) / (total_scenes * 2))
        
        prompts.sort(key=lambda x: x[0])
        
        # 3. 이미지 생성 (선택된 모델 사용)
        status_box.write(f"🎨 이미지 생성 중 ({SELECTED_IMAGE_MODEL})...")
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {}
            for s_num, prompt_text in prompts:
                idx = s_num - 1
                orig_text = chunks[idx]
                fname = make_filename(s_num, orig_text)
                
                # ★ generate_image 함수에 SELECTED_IMAGE_MODEL 전달
                future = executor.submit(generate_image, client, prompt_text, fname, IMAGE_OUTPUT_DIR, SELECTED_IMAGE_MODEL)
                future_to_meta[future] = (s_num, fname, orig_text, prompt_text)
            
            completed_cnt = 0
            for future in as_completed(future_to_meta):
                s_num, fname, orig_text, p_text = future_to_meta[future]
                path = future.result()
                if path:
                    results.append({
                        "scene": s_num,
                        "path": path,
                        "filename": fname,
                        "script": orig_text,
                        "prompt": p_text
                    })
                completed_cnt += 1
                progress_bar.progress(0.5 + (completed_cnt / total_scenes * 0.5))
        
        results.sort(key=lambda x: x['scene'])
        st.session_state['generated_results'] = results
        
        status_box.update(label="✅ 완료되었습니다!", state="complete", expanded=False)
        st.session_state['is_processing'] = False

# ==========================================
# [UI] 결과창
# ==========================================
if st.session_state['generated_results']:
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"📸 결과물 ({len(st.session_state['generated_results'])}장)")
    with col2:
        zip_data = create_zip_buffer(IMAGE_OUTPUT_DIR)
        st.download_button(
            label="📦 전체 ZIP 다운로드",
            data=zip_data,
            file_name="all_images.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    for item in st.session_state['generated_results']:
        with st.container(border=True):
            cols = st.columns([1, 2])
            
            with cols[0]:
                try:
                    st.image(item['path'], use_container_width=True)
                except:
                    st.error("이미지 없음")
            
            with cols[1]:
                st.subheader(f"Scene {item['scene']:02d}")
                st.caption(f"파일명: {item['filename']}")
                st.write(f"**대본:** {item['script']}")
                
                try:
                    with open(item['path'], "rb") as file:
                        btn = st.download_button(
                            label="⬇️ 저장",
                            data=file,
                            file_name=item['filename'],
                            mime="image/png",
                            key=f"btn_down_{item['scene']}"
                        )
                except Exception:
                    st.error("파일 오류")
