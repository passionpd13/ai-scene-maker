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
st.set_page_config(page_title="AI 유튜브 롱폼 제작기 (Pro)", layout="wide", page_icon="🎬")

# 파일 저장 경로 설정
BASE_PATH = "./web_result_files"
IMAGE_OUTPUT_DIR = os.path.join(BASE_PATH, "output_images")

# 텍스트 모델 설정
GEMINI_TEXT_MODEL_NAME = "gemini-2.5-pro"

# ==========================================
# [함수] 1. 대본 구조화 로직
# ==========================================
def generate_structure(client, full_script):
    """Gemini를 이용해 대본 구조화"""
    prompt = f"""
    [Role]
    You are a professional YouTube Content Editor and Scriptwriter.

    [Task]
    Analyze the provided transcript (script).
    Restructure the content into a highly detailed, list-style format suitable for a blog post or a new video plan.
    
    [Output Format]
    1. **Video Theme/Title**: (Extract or suggest a catchy title based on the whole script)
    2. **Intro**: (Hook and background, no music) Approve specific channel names, The intro hooks the overall topic
    3. **Chapter 1** to **Chapter 8**: (Divide the main content into logical sections. Use detailed bullet points for each chapter.)
    4. **Epilogue**: (Conclusion and Subscribe Like Comments that make you anticipate the next specific content)

    [Constraint]
    - Analyze the entire context deeply.
    - Write the output in **Korean**.
    - Make the content rich and detailed.
    - If the original script has a channel name, remove it.

    [Transcript]
    {full_script}
    """
    
    try:
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# [함수] 2. 섹션별 대본 생성 (병렬/개별 공용)
# ==========================================
def generate_section(client, section_title, full_structure, duration_type="fixed"):
    """
    duration_type: '2min', '3min', '4min', 'fixed'(Intro/Epilogue용)
    """
    
    # 1. 분량에 따른 글자수 및 지침 설정
    if duration_type == "2min":
        target_chars = "약 1,100자 (공백 포함)"
        detail_level = "핵심 내용 위주로 명확하게 전달하되, 너무 짧지 않게 서술하십시오."
    elif duration_type == "3min":
        target_chars = "약 1,600자 (공백 포함)"
        detail_level = "충분한 예시와 설명을 곁들여 상세하게 서술하십시오."
    elif duration_type == "4min":
        target_chars = "약 2,100자 이상 (공백 포함)"
        detail_level = "현미경으로 들여다보듯 아주 깊이 있고 디테일하게 묘사하십시오. 절대 요약하지 마십시오."
    else: # Intro / Epilogue (Fixed)
        target_chars = "약 400단어 (약 1,400자)"
        detail_level = "시청자를 사로잡는 강력한 후킹과 여운을 주는 마무리로 작성하십시오. 안녕 인사는 하지 않는다"

    prompt = f"""
    [Role]
    당신은 대한민국 최고의 유튜브 다큐멘터리 작가입니다.

    [Task]
    전체 대본 구조 중 오직 **"{section_title}"** 부분만 작성하십시오.
    
    [Context (Overall Structure)]
    {full_structure}

    [Target Section]
    **{section_title}**

    [Length Constraints]
    - **목표 분량: {target_chars}** - **작성 지침:** {detail_level}
    
    [Style Guidelines]
    1. '습니다' 체를 사용하고, 다큐멘터리 특유의 진지하고 몰입감 있는 어조를 유지하세요.
    2. 앞뒤 문맥(이전 챕터, 다음 챕터)을 고려하되, 이 파트의 내용에만 집중하세요.
    3. (지문), (효과음) 같은 연출 지시어는 제외하고 **오직 나레이션 대사만** 출력하세요.
    4. 서두에 "네, 알겠습니다" 같은 잡담을 하지 말고 바로 대본 내용을 시작하세요.
    5. "영문 병기(English parallel notation)는 생략해 주세요." "영문 병기는 삭제해 주세요." "스크립트 작성할 때 '빅휠(Big Wheel)'처럼 괄호 넣지 말고, 그냥 깔끔하게 '빅휠'로 한글만 표기해 줘."
    6. 쉼표와 접속어 등을 사용하여, 리듬이 있지만 너무 끊기지 않는 흐름을 만들 것
       ❌ "당신은 말에 묶입니다. 말이 달립니다. 살점이 찢깁니다." 이렇게 문장을 너무 짧게 하지 말것
       ✅ "당신은 말의 뒷다리에 포박된 채 끌려가며, 돌과 자갈 위를 미끄러지듯 지나가다 결국 살점이 찢기고 뼈가 드러납니다."

    [Output]
    (지금 바로 {section_title}의 원고를 작성 시작하세요)
    """
    
    try:
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL_NAME, 
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.75 
            )
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# [함수] 3. 이미지 생성 관련 로직
# ==========================================

def init_folders():
    if os.path.exists(IMAGE_OUTPUT_DIR):
        try:
            shutil.rmtree(IMAGE_OUTPUT_DIR)
        except Exception:
            pass
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

def split_script_by_time(script, chars_per_chunk=100):
    # 문장 단위로 쪼개서 청크 만들기
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

# [중요 수정] 전체 제목(video_title)을 인자로 받도록 수정
def generate_prompt(api_key, index, text_chunk, style_instruction, video_title):
    scene_num = index + 1
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TEXT_MODEL_NAME}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}

    # [중요 수정] 프롬프트 상단에 전체 제목을 맥락으로 제공
    full_instruction = f"""
    [Role]
    You are an expert AI art director.

    [Overall Context / Video Title]
    "{video_title}"
    (All images must align with this overall theme and atmosphere.)

    [Style Guideline]
    {style_instruction}

    [Task]
    Create a detailed image generation prompt based on the provided script chunk.
    Describe the scene visually in English. Focus on the visual elements described in the Style Guideline.
    Ensure the scene is consistent with the [Overall Context].
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
    full_path = os.path.join(output_dir, filename)
    try:
        # [사용자 요청] Gemini 이미지 생성 Config 사용
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
        print(f"이미지 생성 에러: {e}")
        return None

def create_zip_buffer(source_dir):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.basename(file_path))
    buffer.seek(0)
    return buffer

# ==========================================
# [UI] 사이드바
# ==========================================
with st.sidebar:
    st.header("⚙️ 환경 설정")
    api_key = st.text_input("🔑 Google API Key : AIzaSyD72pizO1rOmv2Vrl_YpWlLtGRyeM19ZmA", type="password", help="Gemini API 키를 입력하세요.")
    st.markdown("---")
    
    st.subheader("🖼️ 이미지 모델 선택")
    # [사용자 지정 로직 적용]
    model_choice = st.radio("사용할 AI 모델:", ("Premium (Gemini 3 Pro)", "Fast (Gemini 2.5 Flash)"), index=0)
    
    if "Gemini 3 Pro" in model_choice:
        SELECTED_IMAGE_MODEL = "gemini-3-pro-image-preview"
    else:
        SELECTED_IMAGE_MODEL = "gemini-2.5-flash-image"

    st.info(f"✅ 선택 모델: `{SELECTED_IMAGE_MODEL}`")
    
    st.markdown("---")
    st.subheader("⏱️ 장면 분할 설정")
    chunk_duration = st.slider("한 장면당 지속 시간 (초)", 10, 60, 20, 5)
    chars_limit = chunk_duration * 8 
    
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
# [UI] 메인 화면 1: 대본 구조화 및 생성
# ==========================================
st.title("📺 AI 롱폼 영상 제작기 (Pro)")
st.caption("구조 분석 ➡️ 롱폼 대본 생성(병렬 처리) ➡️ 이미지 생성까지 한번에!")

# 세션 초기화
if 'structured_content' not in st.session_state:
    st.session_state['structured_content'] = None
if 'section_scripts' not in st.session_state:
    st.session_state['section_scripts'] = {}
# [신규] 영상 제목 저장을 위한 세션 변수
if 'video_title' not in st.session_state:
    st.session_state['video_title'] = ""

# 1. 구조 분석 섹션
with st.container(border=True):
    raw_script = st.text_area("✍️ 분석할 원고(대본)를 여기에 붙여넣으세요:", height=200, placeholder="안녕하세요, 오늘은...")
    analyze_btn = st.button("🔍 구조 분석 실행", width="stretch", type="primary")

    if analyze_btn:
        if not api_key:
            st.error("⚠️ 사이드바에서 API Key를 먼저 입력해주세요.")
        elif not raw_script:
            st.warning("⚠️ 분석할 대본 내용이 없습니다.")
        else:
            client = genai.Client(api_key=api_key)
            with st.status("대본 내용 분석 중...", expanded=True) as status:
                status.write(f"🧠 Gemini가 내용을 읽고 구조를 잡고 있습니다...")
                result_text = generate_structure(client, raw_script)
                
                st.session_state['structured_content'] = result_text
                st.session_state['section_scripts'] = {} # 구조 바뀌면 대본 초기화

                # [중요] 구조 분석 결과에서 제목 추출 시도 (정규표현식)
                # 예: "1. **Video Theme/Title**: 실제 제목 내용" 형식 찾기
                title_match = re.search(r'^\s*1\.\s*\*\*(.*?)\*\*:\s*(.*)', result_text, re.MULTILINE)
                if title_match:
                    # 콜론 뒤의 내용이 있으면 그것을, 없으면 괄호 안의 내용을 제목으로 사용
                    extracted_title = title_match.group(2).strip() if title_match.group(2).strip() else title_match.group(1).strip()
                    # 혹시 모를 괄호 설명 제거
                    extracted_title = re.sub(r'\(.*?\)', '', extracted_title).strip()
                    st.session_state['video_title'] = extracted_title
                else:
                    st.session_state['video_title'] = "제목을 찾을 수 없음 (직접 입력해주세요)"

                status.update(label="✅ 분석 완료! 제목이 추출되었습니다.", state="complete", expanded=False)

# 2. 대본 생성 섹션
if st.session_state['structured_content']:
    st.divider()
    st.subheader("📑 대본 구조화 결과")
    st.markdown(st.session_state['structured_content'])
    
    # [신규] 추출된 제목 확인 및 수정 UI
    st.info(f"📌 **추출된 영상 제목:** {st.session_state['video_title']} (이미지 생성 단계에서 수정 가능)")

    st.divider()
    st.subheader("⚡ 롱폼 대본 전체 일괄 생성 (병렬 처리)")
    st.caption("🚀 버튼 한번으로 모든 챕터를 동시에 작성합니다. (15분/20분/25분 옵션)")

    # 챕터 목록 추출
    lines = st.session_state['structured_content'].split('\n')
    chapter_titles = ["Intro (도입부)"]
    found_chapters = re.findall(r'(?:Chapter|챕터)\s*\d+.*', st.session_state['structured_content'])
    seen = set()
    for ch in found_chapters:
        clean_ch = ch.replace('*', '').strip()
        if clean_ch not in seen:
            chapter_titles.append(clean_ch)
            seen.add(clean_ch)
    chapter_titles.append("Epilogue (결론)")
    
    for title in chapter_titles:
        if title not in st.session_state['section_scripts']:
            st.session_state['section_scripts'][title] = ""

    # 일괄 생성 패널
    with st.container(border=True):
        col_batch1, col_batch2 = st.columns([1, 1])
        with col_batch1:
            target_time = st.radio(
                "🎬 총 영상 목표 시간 (텍스트 분량)",
                ("15분 (약 7,000자)", "20분 (약 10,000자)", "25분 (약 13,000자)"),
                index=1
            )
            if "15분" in target_time: batch_duration_type = "2min" 
            elif "20분" in target_time: batch_duration_type = "3min" 
            else: batch_duration_type = "4min"

        with col_batch2:
            st.write("")
            st.write("") 
            batch_btn = st.button("🚀 전체 대본 동시 생성 시작", type="primary", use_container_width=True)

    if batch_btn:
        if not api_key:
            st.error("⚠️ API Key가 필요합니다.")
        else:
            client = genai.Client(api_key=api_key)
            status_box = st.status("🚀 AI가 모든 챕터를 동시에 작성 중입니다...", expanded=True)
            progress_bar = status_box.progress(0)
            
            total_tasks = len(chapter_titles)
            completed_tasks = 0
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_title = {}
                for title in chapter_titles:
                    is_fixed = any(x in title for x in ["Intro", "Epilogue", "도입부", "결론"])
                    current_duration = "fixed" if is_fixed else batch_duration_type
                    future = executor.submit(generate_section, client, title, st.session_state['structured_content'], current_duration)
                    future_to_title[future] = title
                
                for future in as_completed(future_to_title):
                    title = future_to_title[future]
                    try:
                        result_text = future.result()
                        st.session_state['section_scripts'][title] = result_text
                        st.session_state[f"txt_{title}"] = result_text # 화면 동기화
                        completed_tasks += 1
                        progress_bar.progress(completed_tasks / total_tasks)
                        status_box.write(f"✅ 완료: {title}")
                    except Exception as e:
                        status_box.error(f"❌ 실패 ({title}): {e}")
            
            status_box.update(label="✨ 전체 생성 완료! 아래에서 확인하세요.", state="complete", expanded=False)
            time.sleep(1)
            st.rerun()

    # 섹션별 확인 및 수정
    st.subheader("📝 섹션별 확인 및 수정")
    full_combined_script = ""
    
    for title in chapter_titles:
        with st.expander(f"📌 {title}", expanded=False):
            is_intro_epilogue = any(x in title for x in ["Intro", "Epilogue", "도입부", "결론"])
            
            if is_intro_epilogue:
                if st.button(f"🔄 {title} 다시 생성", key=f"r_fix_{title}"):
                    client = genai.Client(api_key=api_key)
                    with st.spinner("재생성 중..."):
                        result = generate_section(client, title, st.session_state['structured_content'], "fixed")
                        st.session_state['section_scripts'][title] = result
                        st.session_state[f"txt_{title}"] = result 
                        st.rerun()
            else:
                c_cols = st.columns(3)
                def regen(dur):
                    client = genai.Client(api_key=api_key)
                    with st.spinner(f"{dur} 모드로 재생성 중..."):
                        dur_code = "2min" if "2분" in dur else "3min" if "3분" in dur else "4min"
                        result = generate_section(client, title, st.session_state['structured_content'], dur_code)
                        st.session_state['section_scripts'][title] = result
                        st.session_state[f"txt_{title}"] = result
                        st.rerun()

                if c_cols[0].button("🔄 다시 생성 (2분)", key=f"r2_{title}"): regen("2분")
                if c_cols[1].button("🔄 다시 생성 (3분)", key=f"r3_{title}"): regen("3분")
                if c_cols[2].button("🔄 다시 생성 (4분)", key=f"r4_{title}"): regen("4분")

            if f"txt_{title}" not in st.session_state:
                st.session_state[f"txt_{title}"] = st.session_state['section_scripts'].get(title, "")

            new_text = st.text_area(label="📜 대본 내용 (수정 가능)", height=300, key=f"txt_{title}")
            st.session_state['section_scripts'][title] = new_text
        
        if st.session_state['section_scripts'].get(title):
            full_combined_script += st.session_state['section_scripts'][title] + "\n\n"

    # 최종 병합 결과
    if full_combined_script:
        st.divider()
        st.subheader("📦 최종 완성 대본")
        col_info, col_down = st.columns([3, 1])
        with col_info:
            st.caption(f"📝 총 글자 수: {len(full_combined_script)}자 (공백 포함)")
        with col_down:
            st.download_button(label="💾 대본 다운로드 (.txt)", data=full_combined_script, file_name="final_script.txt", mime="text/plain", use_container_width=True)
        st.text_area("아래 내용을 복사하거나 위 버튼을 눌러 저장하세요", value=full_combined_script, height=500)

# ==========================================
# [UI] 메인 화면 3: 이미지 생성
# ==========================================
st.divider()
st.title("🎬 AI 씬(장면) 생성기 (Pro)")
st.caption(f"완성된 대본을 넣으면 장면별 이미지를 생성합니다. | 🎨 Model: {SELECTED_IMAGE_MODEL}")

# [중요 추가] 전체 영상 제목 입력/수정란
st.subheader("📌 전체 영상 테마(제목) 설정")
st.caption("이미지 생성 시 이 제목이 '전체적인 분위기 기준'이 됩니다. 필요하면 수정하세요.")
video_title_input = st.text_input(
    "영상 제목 (여기에 입력된 내용이 프롬프트의 기준이 됩니다)",
    value=st.session_state['video_title'], # 구조 분석에서 추출한 값이 기본값
    key="video_title_input_field"
)
# 입력된 값을 세션에 업데이트
st.session_state['video_title'] = video_title_input


# 대본 자동 분류 및 가져오기 기능
if 'section_scripts' in st.session_state and st.session_state['section_scripts']:
    intro_text_acc = ""
    main_text_acc = ""
    for title, text in st.session_state['section_scripts'].items():
        if "Intro" in title or "도입부" in title:
            intro_text_acc += text + "\n\n"
        else:
            main_text_acc += text + "\n\n"
            
    st.write("👇 **생성된 대본 가져오기 (클릭 시 아래 입력창에 채워집니다)**")
    col_get1, col_get2 = st.columns(2)
    if "image_gen_input" not in st.session_state:
        st.session_state["image_gen_input"] = ""

    with col_get1:
        if st.button("📥 인트로(Intro)만 가져오기", use_container_width=True):
            st.session_state["image_gen_input"] = intro_text_acc.strip()
            st.rerun()
    with col_get2:
        if st.button("📥 본론(Chapters) + 결론(Epilogue) 가져오기", use_container_width=True):
            st.session_state["image_gen_input"] = main_text_acc.strip()
            st.rerun()

script_input = st.text_area(
    "📜 이미지로 만들 대본 입력", 
    height=300, 
    placeholder="위 버튼을 눌러 대본을 가져오거나, 직접 붙여넣으세요...",
    key="image_gen_input"
)

if 'generated_results' not in st.session_state:
    st.session_state['generated_results'] = []
if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False

start_btn = st.button("🚀 이미지 생성 시작", type="primary", width="stretch")

if start_btn:
    if not api_key:
        st.error("⚠️ API Key를 입력해주세요.")
    elif not script_input:
        st.warning("⚠️ 대본을 입력해주세요.")
    elif not st.session_state['video_title']:
        st.warning("⚠️ 영상 제목을 설정해주세요. (이미지 생성의 기준이 됩니다)")
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
        
        # 2. 프롬프트 생성 (병렬) - [중요] video_title 인자 전달
        status_box.write(f"📝 프롬프트 작성 중 ({GEMINI_TEXT_MODEL_NAME}) - 기준 테마: {st.session_state['video_title']}...")
        prompts = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            current_video_title = st.session_state['video_title'] # 현재 입력된 제목 사용
            for i, chunk in enumerate(chunks):
                # generate_prompt 함수에 video_title 전달
                futures.append(executor.submit(generate_prompt, api_key, i, chunk, style_instruction, current_video_title))
            
            for i, future in enumerate(as_completed(futures)):
                prompts.append(future.result())
                progress_bar.progress((i + 1) / (total_scenes * 2))
        
        prompts.sort(key=lambda x: x[0])
        
        # 3. 이미지 생성 (병렬)
        status_box.write(f"🎨 이미지 생성 중 ({SELECTED_IMAGE_MODEL})...")
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {}
            for s_num, prompt_text in prompts:
                idx = s_num - 1
                orig_text = chunks[idx]
                fname = make_filename(s_num, orig_text)
                
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

# 결과창
if st.session_state['generated_results']:
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"📸 결과물 ({len(st.session_state['generated_results'])}장)")
    with col2:
        zip_data = create_zip_buffer(IMAGE_OUTPUT_DIR)
        st.download_button("📦 전체 ZIP 다운로드", data=zip_data, file_name="all_images.zip", mime="application/zip", use_container_width=True)
    
    for item in st.session_state['generated_results']:
        with st.container(border=True):
            cols = st.columns([1, 2])
            with cols[0]:
                try: st.image(item['path'], use_container_width=True)
                except: st.error("이미지 없음")
            with cols[1]:
                st.subheader(f"Scene {item['scene']:02d}")
                st.caption(f"파일명: {item['filename']}")
                st.write(f"**대본:** {item['script']}")
                with st.expander("프롬프트 확인"):
                    st.text(item['prompt'])
                try:
                    with open(item['path'], "rb") as file:
                        st.download_button("⬇️ 저장", data=file, file_name=item['filename'], mime="image/png", key=f"btn_down_{item['scene']}")
                except: st.error("파일 오류")

