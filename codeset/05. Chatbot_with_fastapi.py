from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import os
import requests
import openai
import ast
import math
from dotenv import load_dotenv
load_dotenv()

VECTORDB_PATH = os.getenv('VECTORDB_PATH', './vectordb')
if not os.path.isabs(VECTORDB_PATH):
    VECTORDB_PATH = os.path.abspath(VECTORDB_PATH)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')

# ChromaDB 1.x 방식: PersistentClient 사용
client = chromadb.PersistentClient(path=VECTORDB_PATH)

app = FastAPI()

class ReportRequest(BaseModel):
    member_id: str
    llm: str  # 'gpt' or 'ollama'

def safe_eval(doc):
    try:
        return eval(doc, {"nan": None, "None": None, "null": None, "NaN": None, "inf": None, "Infinity": None, "true": True, "false": False})
    except Exception:
        try:
            return ast.literal_eval(doc)
        except Exception:
            return {}

def get_latest_spo2(member_id):
    # 컬렉션 존재 여부 확인
    collections = [c.name for c in client.list_collections()]
    if 'spo2' not in collections:
        return None
    collection = client.get_collection('spo2')
    results = collection.get(where={"memb_id": member_id})
    docs = results['documents']
    if not docs:
        return None
    docs_sorted = sorted(docs, key=lambda d: safe_eval(d).get('create_dttm', ''), reverse=True)
    latest = safe_eval(docs_sorted[0])
    return latest

def get_latest_blood_pressure(member_id):
    collections = [c.name for c in client.list_collections()]
    if 'blood_pressure' not in collections:
        return None
    collection = client.get_collection('blood_pressure')
    results = collection.get(where={"memb_id": member_id})
    docs = results['documents']
    if not docs:
        return None
    docs_sorted = sorted(docs, key=lambda d: safe_eval(d).get('create_dttm', ''), reverse=True)
    latest = safe_eval(docs_sorted[0])
    return latest

def get_latest_heart_rate(member_id):
    collections = [c.name for c in client.list_collections()]
    if 'heart_rate' not in collections:
        return None
    collection = client.get_collection('heart_rate')
    results = collection.get(where={"memb_id": member_id})
    docs = results['documents']
    if not docs:
        return None
    docs_sorted = sorted(docs, key=lambda d: safe_eval(d).get('create_dttm', ''), reverse=True)
    latest = safe_eval(docs_sorted[0])
    return latest

def get_latest_hdl_cholesterol(member_id):
    collections = [c.name for c in client.list_collections()]
    if 'hdl_cholesterol' not in collections:
        return None
    collection = client.get_collection('hdl_cholesterol')
    results = collection.get(where={"memb_id": member_id})
    docs = results['documents']
    if not docs:
        return None
    docs_sorted = sorted(docs, key=lambda d: safe_eval(d).get('create_dttm', ''), reverse=True)
    latest = safe_eval(docs_sorted[0])
    return latest

def get_latest_ldl_cholesterol(member_id):
    collections = [c.name for c in client.list_collections()]
    if 'ldl_cholesterol' not in collections:
        return None
    collection = client.get_collection('ldl_cholesterol')
    results = collection.get(where={"memb_id": member_id})
    docs = results['documents']
    if not docs:
        return None
    docs_sorted = sorted(docs, key=lambda d: safe_eval(d).get('create_dttm', ''), reverse=True)
    latest = safe_eval(docs_sorted[0])
    return latest

# 프롬프트 엔지니어링 보강: 데이터가 없으면 '저장된 측정값이 없음'으로 표기
def val_or_none(val, unit="", data_type=""):
    # None, NaN, 빈 문자열, 'nan', 'None', 'null', 0 등 모두 체크
    if val is None:
        return f"{data_type} 측정값이 없습니다. 기기 연동 후 업데이트해주세요."
    if isinstance(val, str) and (val.strip() == "" or val.strip().lower() in ["none", "nan", "null"]):
        return f"{data_type} 측정값이 없습니다. 기기 연동 후 업데이트해주세요."
    try:
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return f"{data_type} 측정값이 없습니다. 기기 연동 후 업데이트해주세요."
    except Exception:
        pass
    return f"{val}{unit}"

def build_prompt(latest_glucose, latest_spo2, latest_bp, latest_hr, latest_hdl, latest_ldl):
    return f"""
[혈당 결과 해석]
- 혈당: {val_or_none(latest_glucose.get('glucosedata') if latest_glucose else None, 'mg/dL', '혈당')} (측정일시: {val_or_none(latest_glucose.get('get_time') if latest_glucose else None, '', '혈당 측정 시간')})
- 위 수치가 정상/경계/위험 중 어디에 해당하는지, 그 이유와 함께 설명하라.

[산소포화도 결과 해석]
- 산소포화도: {val_or_none(latest_spo2.get('spo2') if latest_spo2 else None, '%', '산소포화도')} (측정일시: {val_or_none(latest_spo2.get('get_time') if latest_spo2 else None, '', '산소포화도 측정 시간')})
- 위 수치가 정상/경계/위험 중 어디에 해당하는지, 그 이유와 함께 설명하라.

[혈압 결과 해석]
- 혈압: {val_or_none(latest_bp.get('systolic') if latest_bp else None, '', '수축기 혈압')}/{val_or_none(latest_bp.get('diastolic') if latest_bp else None, '', '이완기 혈압')} mmHg (측정일시: {val_or_none(latest_bp.get('get_time') if latest_bp else None, '', '혈압 측정 시간')})
- 위 수치가 정상/경계/위험 중 어디에 해당하는지, 그 이유와 함께 설명하라.

[심박수 결과 해석]
- 심박수: {val_or_none(latest_hr.get('heart_rate') if latest_hr else None, '회/분', '심박수')} (측정일시: {val_or_none(latest_hr.get('get_time') if latest_hr else None, '', '심박수 측정 시간')})
- 위 수치가 정상/경계/위험 중 어디에 해당하는지, 그 이유와 함께 설명하라.

[콜레스테롤 결과 해석]
- HDL: {val_or_none(latest_hdl.get('cho_value') if latest_hdl else None, 'mg/dL', 'HDL 콜레스테롤')} (측정일시: {val_or_none(latest_hdl.get('get_time') if latest_hdl else None, '', 'HDL 측정 시간')})
- LDL: {val_or_none(latest_ldl.get('cho_value') if latest_ldl else None, 'mg/dL', 'LDL 콜레스테롤')} (측정일시: {val_or_none(latest_ldl.get('get_time') if latest_ldl else None, '', 'LDL 측정 시간')})
- 위 수치가 정상/경계/위험 중 어디에 해당하는지, 그 이유와 함께 설명하라.

[식사 진단]
- 위의 건강 지표(혈당, 콜레스테롤, 혈압 등) 상태를 바탕으로, 식사에서 주의해야 할 점이나 추천할 점을 간단히 제안하라.

[운동 진단]
- 위의 건강 지표 상태를 바탕으로, 운동에서 주의해야 할 점이나 추천할 점을 간단히 제안하라.

[지침]
- 위의 모든 섹션(혈당, 산소포화도, 혈압, 심박수, 콜레스테롤, 식사, 운동, 지침)에 대해 반드시 답변하라.
- 각 지표의 측정일자와 값을 명확히 언급하라.
- 각 수치가 정상/경계/위험 중 어디에 해당하는지, 그 이유와 함께 설명하라.
- 식사/운동/지침은 현재 건강 지표 상태에 맞는 맞춤형 가이드로 제안하라.
- 데이터가 없는 경우 해당 지표의 중요성을 설명하고, 기기 연동을 통해 측정 데이터를 추가하도록 안내하라.

[예시]
- 2024-03-09 20:24:40에 측정한 HDL 콜레스테롤 수치는 42mg/dL로 경계 범위에 해당합니다. 경계 범위에 해당하므로 식사에서 지방 섭취를 줄이고, 운동을 꾸준히 하는 것이 좋습니다.
"""

def get_latest_blood_glucose(member_id):
    # 컬렉션 존재 여부 확인
    collections = [c.name for c in client.list_collections()]
    if 'blood_glucose' not in collections:
        raise HTTPException(status_code=404, detail="[blood_glucose] collection이 존재하지 않습니다. 먼저 데이터 적재를 진행하세요.")
    collection = client.get_collection('blood_glucose')
    results = collection.get(where={"memb_id": member_id})
    docs = results['documents']
    if not docs:
        raise HTTPException(status_code=404, detail="해당 member_id로 적재된 혈당 데이터가 없습니다.")
    docs_sorted = sorted(docs, key=lambda d: safe_eval(d).get('create_dttm', ''), reverse=True)
    latest = safe_eval(docs_sorted[0])
    return latest

def call_openai(prompt):
    print(f"[INFO] OpenAI API 호출 시작...")
    openai.api_key = OPENAI_API_KEY
    system_message = "당신은 파밀리케어 담당 건강관리 챗봇입니다. 마지막에는 파밀리케어 답변이었습니다로 끝내줘."
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        result = response.choices[0].message.content.strip()
        print(f"[INFO] OpenAI API 호출 성공 (응답 길이: {len(result)} 문자)")
        return result
    except Exception as e:
        print(f"[ERROR] OpenAI API 호출 실패: {str(e)}")
        raise

def call_ollama(prompt):
    print(f"[INFO] Ollama API 호출 시작... (URL: {OLLAMA_URL})")
    system_message = "당신은 파밀리케어 담당 건강관리 챗봇입니다. 마지막에는 파밀리케어 답변이었습니다로 끝내줘."
    full_prompt = f"<|system|>{system_message}\n<|user|>{prompt}"
    try:
        print(f"[INFO] Ollama 요청 전송 중... (timeout: 180초)")
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "gemma3:4b", "prompt": full_prompt, "stream": False},
            timeout=180  # 3분 timeout
        )
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            print(f"[INFO] Ollama API 호출 성공 (응답 길이: {len(result)} 문자)")
            return result
        else:
            error_msg = f"[오류] Ollama 호출 실패: {response.text}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    except requests.exceptions.Timeout:
        error_msg = "[오류] Ollama 호출 타임아웃 (180초 초과)"
        print(f"[ERROR] {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"[오류] Ollama 호출 실패: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg

@app.post("/generate_report")
def generate_report(req: ReportRequest):
    print(f"[INFO] POST 요청 수신: member_id={req.member_id}, llm={req.llm}")
    
    try:
        print(f"[INFO] 벡터 DB에서 건강 데이터 조회 시작...")
        latest_glucose = get_latest_blood_glucose(req.member_id)
        latest_spo2 = get_latest_spo2(req.member_id)
        latest_bp = get_latest_blood_pressure(req.member_id)
        latest_hr = get_latest_heart_rate(req.member_id)
        latest_hdl = get_latest_hdl_cholesterol(req.member_id)
        latest_ldl = get_latest_ldl_cholesterol(req.member_id)
        
        print(f"[INFO] 데이터 조회 완료 - 혈당: {'있음' if latest_glucose else '없음'}, 산소포화도: {'있음' if latest_spo2 else '없음'}, 혈압: {'있음' if latest_bp else '없음'}, 심박수: {'있음' if latest_hr else '없음'}, HDL: {'있음' if latest_hdl else '없음'}, LDL: {'있음' if latest_ldl else '없음'}")
        
        if not latest_glucose:
            print(f"[ERROR] 혈당 데이터 없음: {req.member_id}")
            raise HTTPException(status_code=404, detail="해당 member_id로 적재된 혈당 데이터가 없습니다.")
        
        print(f"[INFO] 프롬프트 생성 시작...")
        prompt = build_prompt(latest_glucose, latest_spo2, latest_bp, latest_hr, latest_hdl, latest_ldl)
        print(f"[INFO] 프롬프트 생성 완료 (길이: {len(prompt)} 문자)")
        
        print(f"[INFO] LLM 호출 시작: {req.llm}")
        if req.llm == 'gpt':
            result = call_openai(prompt)
            print(f"[INFO] OpenAI GPT 호출 완료")
        elif req.llm == 'ollama':
            result = call_ollama(prompt)
            print(f"[INFO] Ollama 호출 완료")
        else:
            print(f"[ERROR] 지원하지 않는 LLM: {req.llm}")
            raise HTTPException(status_code=400, detail="llm 파라미터는 'gpt' 또는 'ollama'만 지원합니다.")
        
        print(f"[INFO] 응답 생성 완료 (길이: {len(result)} 문자)")
        return {"member_id": req.member_id, "llm": req.llm, "result": result}
        
    except Exception as e:
        print(f"[ERROR] 요청 처리 중 오류 발생: {str(e)}")
        raise

# ollama 사용 예시 (Postman 등에서)
# {
#   "member_id": "52a62c11-7c4f-4912-91c9-ff5c145328cd",
#   "llm": "ollama"
# }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("FaMiliCare Health Chatbot API 서버를 시작합니다...")
    print(f"벡터 DB 경로: {VECTORDB_PATH}")
    print(f"OpenAI API 키: {'설정됨' if OPENAI_API_KEY else '설정되지 않음'}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"서버 주소: http://0.0.0.0:9000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=9000) 