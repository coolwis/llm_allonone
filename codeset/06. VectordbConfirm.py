import os
import chromadb
from dotenv import load_dotenv
load_dotenv()

VECTORDB_PATH = os.getenv('VECTORDB_PATH', './vectordb')
if not os.path.isabs(VECTORDB_PATH):
    VECTORDB_PATH = os.path.abspath(VECTORDB_PATH)

client = chromadb.PersistentClient(path=VECTORDB_PATH)

# 컬렉션 리스트 조회
def list_collections():
    collections = [c.name for c in client.list_collections()]
    print(f"[INFO] 현재 벡터DB에 존재하는 컬렉션: {collections}")
    return collections

# 각 컬렉션에서 데이터 일부 샘플 조회
def show_sample_from_collection(collection_name, n=3):
    if collection_name not in [c.name for c in client.list_collections()]:
        print(f"[WARN] 컬렉션 '{collection_name}'이 존재하지 않습니다.")
        return
    collection = client.get_collection(collection_name)
    results = collection.get()
    print(f"[INFO] '{collection_name}' 내 데이터 개수: {len(results['ids'])}")
    for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
        if i >= n:
            break
        print(f"[{i+1}] doc: {doc}\n    meta: {meta}")

if __name__ == "__main__":
    # 컬렉션 리스트 확인
    collections = list_collections()
    # 각 컬렉션에서 샘플 데이터 3개씩 출력
    for cname in collections:
        show_sample_from_collection(cname, n=3) 