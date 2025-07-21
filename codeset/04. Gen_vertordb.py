import os
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

# 환경변수 로드
load_dotenv()
DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT'))
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
# VECTORDB_PATH 환경변수에서 읽고, 상대경로면 절대경로로 변환
VECTORDB_PATH = os.getenv('VECTORDB_PATH', './vectordb')
if not os.path.isabs(VECTORDB_PATH):
    VECTORDB_PATH = os.path.abspath(VECTORDB_PATH)
try:
    os.makedirs(VECTORDB_PATH, exist_ok=True)
    print(f"[DEBUG] VECTORDB_PATH: {VECTORDB_PATH}")
    print(f"[DEBUG] 폴더 존재 여부: {os.path.exists(VECTORDB_PATH)}")
    print(f"[DEBUG] 폴더 내 파일: {os.listdir(VECTORDB_PATH)}")
except Exception as e:
    print(f"[ERROR] VECTORDB_PATH 폴더 생성/접근 에러: {e}")
UPDATE_THRESHOLD_HOURS = int(os.getenv('UPDATE_THRESHOLD_HOURS', '24'))

# 1. 임베딩 모델 로드 (한국어 SOTA)
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# 환경변수에서 OpenAI 키를 불러올 수도 있음
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# HuggingFace 임베더는 미리 로드
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def get_hf_embedding(text):
    return embedder.encode(text, convert_to_numpy=True).tolist()

def get_openai_embedding(text, api_key):
    openai.api_key = api_key
    response = openai.embeddings.create(
        input=[text],
        model=OPENAI_EMBED_MODEL
    )
    return response.data[0].embedding

def get_embedding(text, embedding_mode='hf', openai_api_key=None):
    if embedding_mode == 'openai':
        return get_openai_embedding(text, openai_api_key or OPENAI_API_KEY)
    else:
        return get_hf_embedding(text)

def get_db_connection():
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    print("[INFO] DB 접속 완료")
    return conn

def fetch_data(table, columns, since=None):
    conn = get_db_connection()
    query = f"SELECT {', '.join(columns)} FROM {table}"
    if since:
        query += " WHERE create_dttm >= %s"
        df = pd.read_sql(query, conn, params=[since])
    else:
        df = pd.read_sql(query, conn)
    conn.close()
    return df

def safe_meta_value(val):
    # None이면 빈 문자열, 아니면 그대로 반환
    return "" if val is None else val

# ChromaDB 1.x 방식: PersistentClient 사용
client = chromadb.PersistentClient(path=VECTORDB_PATH)

def check_collection_data(collection_name):
    if collection_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(collection_name)
        results = collection.get()
        print(f"[DEBUG] {collection_name} 내 데이터 개수: {len(results['ids'])}")
    else:
        print(f"[DEBUG] {collection_name} 컬렉션이 존재하지 않습니다.")

def save_to_vectordb(
    collection_name, df, text_col, meta_cols, data_type,
    embedding_mode='hf',  # 'hf' or 'openai'
    openai_api_key=None,
    full_reload=False
):
    try:
        print(f"[DEBUG] ChromaDB PersistentClient path: {VECTORDB_PATH}")
        if collection_name not in [c.name for c in client.list_collections()]:
            client.create_collection(collection_name)
        collection = client.get_collection(collection_name)
        if full_reload:
            print(f"[INFO] {collection_name} 전체 재적재: 기존 데이터 삭제 (컬렉션 drop/recreate)")
            client.delete_collection(collection_name)
            client.create_collection(collection_name)
            collection = client.get_collection(collection_name)
        print(f"[DEBUG] 적재할 데이터 {len(df)}건")
        if not df.empty:
            ids = df['id'].astype(str).tolist()
            texts = df[text_col].astype(str).tolist()
            embeddings = [get_embedding(text, embedding_mode, openai_api_key) for text in texts]
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            metadatas = [
                {
                    **{col: safe_meta_value(row[col]) for col in meta_cols},
                    'is_private': True,
                    'data_source': safe_meta_value(row['device_desc']),
                    'data_type': data_type,
                    'created_by_system': 'FaMiliCare',
                    'ingest_time': now,
                    'anonymized': False,
                    'consent_granted': True
                }
                for _, row in df.iterrows()
            ]
            documents = df.to_dict('records')
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=[str(doc) for doc in documents]
            )
            print(f"[INFO] {collection_name} 벡터DB 적재 완료 ({len(df)}건)")
            print(f"[DEBUG] 적재 후 폴더 내 파일: {os.listdir(VECTORDB_PATH)}")
            check_collection_data(collection_name)
        else:
            print(f"[INFO] {collection_name} 적재할 데이터가 없습니다. (DB에서 조회된 데이터가 없습니다)")
    except Exception as e:
        print(f"[ERROR] {collection_name} 적재 중 에러: {e}")

def query_vectordb_with_rerank(collection_name, query_text, text_col, top_k=3, embedding_mode='hf', openai_api_key=None):
    collection = client.get_collection(collection_name)
    # 쿼리 임베딩 생성
    query_embedding = get_embedding(str(query_text), embedding_mode, openai_api_key)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    # ChromaDB 유사도 검색
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=['embeddings', 'documents', 'metadatas']
    )
    docs = results['documents'][0]
    embs = np.array(results['embeddings'][0])
    metas = results['metadatas'][0]
    # scikit-learn cosine similarity로 rerank
    scores = cosine_similarity(query_embedding, embs)[0]
    scored = sorted(zip(docs, metas, embs, scores), key=lambda x: x[3], reverse=True)
    print(f"[INFO] {collection_name} 유사도 기반 top-{top_k} + rerank 결과:")
    for i, (doc, meta, emb, score) in enumerate(scored, 1):
        print(f"[{i}] score={score:.4f} | member_id={meta.get('memb_id')} | doc={doc}")

def list_collections():
    collections = [c.name for c in client.list_collections()]
    print(f"[INFO] 현재 벡터DB에 존재하는 컬렉션: {collections}")
    return collections

# ----------- 혈당 데이터 적재 -----------
blood_glucose_columns = [
    'total_count', 'id', 'memb_id', 'device_desc', 'fullname',
    'get_time', 'glucosedata', 'when_eat', 'create_dttm'
]
since = (datetime.now() - timedelta(hours=UPDATE_THRESHOLD_HOURS)).strftime('%Y-%m-%d %H:%M:%S')
# 전체 적재: since=None, 증분 적재: since=since
MODE = os.getenv('VECTORDB_MODE', 'full')
if MODE == 'full':
    since_param = None
    full_reload = True
else:
    since_param = since
    full_reload = False
df_glucose = fetch_data('fmd_trend_analysis_blood_glucose', blood_glucose_columns, since=since_param)

save_to_vectordb(
    collection_name='blood_glucose',
    df=df_glucose,
    text_col='glucosedata',
    meta_cols=['memb_id', 'device_desc', 'fullname', 'get_time', 'when_eat', 'create_dttm'],
    data_type='blood_glucose',
    embedding_mode='hf',  # 'hf' or 'openai'
    openai_api_key=OPENAI_API_KEY,
    full_reload=full_reload
)

# ----------- 산소포화도 데이터 적재 -----------
spo2_columns = [
    'total_count', 'id', 'memb_id', 'device_desc', 'fullname',
    'get_time', 'spo2', 'heart_rate', 'create_dttm'
]
df_spo2 = fetch_data('fmd_trend_analysis_spo2', spo2_columns, since=since_param)
save_to_vectordb(
    collection_name='spo2',
    df=df_spo2,
    text_col='spo2',
    meta_cols=['memb_id', 'device_desc', 'fullname', 'get_time', 'heart_rate', 'create_dttm'],
    data_type='spo2',
    embedding_mode='hf',  # 'hf' or 'openai'
    openai_api_key=OPENAI_API_KEY,
    full_reload=full_reload
)

blood_pressure_columns = [
    'total_count', 'id', 'memb_id', 'device_desc', 'fullname',
    'get_time', 'systolic', 'diastolic', 'hr', 'create_dttm'
]
df_bp = fetch_data('fmd_trend_analysis_blood_pressure', blood_pressure_columns, since=since_param)
save_to_vectordb(
    collection_name='blood_pressure',
    df=df_bp,
    text_col='systolic',  # 수축기 혈압 기준 벡터화
    meta_cols=['memb_id', 'device_desc', 'fullname', 'get_time', 'diastolic', 'hr', 'create_dttm'],
    data_type='blood_pressure',
    embedding_mode='hf',  # 또는 'openai'
    openai_api_key=OPENAI_API_KEY,
    full_reload=full_reload
)

# heart_rate_columns = [
#     'total_count', 'id', 'memb_id', 'device_desc', 'fullname',
#     'get_time', 'heart_rate', 'create_dttm'
# ]
# df_hr = fetch_data('fmd_trend_analysis_heart_rate', heart_rate_columns, since=since_param)
# save_to_vectordb(
#     collection_name='heart_rate',
#     df=df_hr,
#     text_col='heart_rate',
#     meta_cols=['memb_id', 'device_desc', 'fullname', 'get_time', 'create_dttm'],
#     data_type='heart_rate',
#     embedding_mode='hf',
#     openai_api_key=OPENAI_API_KEY,
#     full_reload=full_reload
# )

hdl_cholesterol_columns = [
    'total_count', 'id', 'memb_id', 'device_desc', 'fullname',
    'get_time', 'cho_value', 'create_dttm'
]
df_hdl = fetch_data('fmd_trend_analysis_hdl_cholesterol', hdl_cholesterol_columns, since=since_param)
save_to_vectordb(
    collection_name='hdl_cholesterol',
    df=df_hdl,
    text_col='cho_value',
    meta_cols=['memb_id', 'device_desc', 'fullname', 'get_time', 'create_dttm'],
    data_type='hdl_cholesterol',
    embedding_mode='hf',
    openai_api_key=OPENAI_API_KEY,
    full_reload=full_reload
)

ldl_cholesterol_columns = [
    'total_count', 'id', 'memb_id', 'device_desc', 'fullname',
    'get_time', 'cho_value', 'create_dttm'
]
df_ldl = fetch_data('fmd_trend_analysis_ldl_cholesterol', ldl_cholesterol_columns, since=since_param)
save_to_vectordb(
    collection_name='ldl_cholesterol',
    df=df_ldl,
    text_col='cho_value',
    meta_cols=['memb_id', 'device_desc', 'fullname', 'get_time', 'create_dttm'],
    data_type='ldl_cholesterol',
    embedding_mode='hf',
    openai_api_key=OPENAI_API_KEY,
    full_reload=full_reload
)

# ----------- 검색 예시 -----------
# 혈당 쿼리 예시 (예: '110')
query_vectordb_with_rerank(
    collection_name='blood_glucose',
    query_text='110',
    text_col='glucosedata',
    top_k=3,
    embedding_mode='hf',  # 'hf' or 'openai'
    openai_api_key=OPENAI_API_KEY
)
# 산소포화도 쿼리 예시 (예: '97')
query_vectordb_with_rerank(
    collection_name='spo2',
    query_text='97',
    text_col='spo2',
    top_k=3,
    embedding_mode='hf',  # 'hf' or 'openai'
    openai_api_key=OPENAI_API_KEY
)

# 사용 예시
if __name__ == "__main__":
    list_collections()
    # ... (기존 적재/검색 코드) ... 