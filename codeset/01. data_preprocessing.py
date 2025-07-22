import os
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 허깅페이스 토큰 및 데이터셋 주소
hf_token = os.getenv('HF_TOKEN')
hf_repo = os.getenv('HF_DATASET_REPO', 'hyokwan/familicare_health_pro_knowledge')
parent_dir = os.getenv('DATASET_PARENT_DIR', '../dataset/01. 필수의학')  # 예: 'dataset/전문의료'

# __file__이 없는 환경(Jupyter 등)에서도 동작하도록 처리
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# 처리할 폴더 리스트 (parent_dir 내 모든 하위 폴더 자동 탐색)
dataset_root = os.path.join(script_dir, parent_dir)

dirs = os.listdir(dataset_root)
data_folders = []
for i in range(len(dirs)):
    d = dirs[i]
    if os.path.isdir(os.path.join(dataset_root, d)):
        data_folders.append(d)

all_rows = []

for i in range(len(data_folders)):
    folder = data_folders[i]
    data_dir = os.path.join(dataset_root, folder)
    if not os.path.exists(data_dir):
        continue
    # 폴더명에서 'TL_' 제거
    input_val = folder.split('_', 1)[-1] if '_' in folder else folder
    files = os.listdir(data_dir)
    json_files = []
    for k in range(len(files)):
        f_name = files[k]
        if f_name.endswith('.json'):
            json_files.append(f_name)
    for j in range(len(json_files)):
        file = json_files[j]
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        # with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
        #     data = json.load(f)
            instruction = data['question']
            output = data['answer']
            all_rows.append({
                'instruction': instruction,
                'input': input_val,
                'output': output
            })

# 데이터프레임 생성
pdf = pd.DataFrame(all_rows)

# Huggingface Datasets 객체로 변환
hf_dataset = Dataset.from_pandas(pdf)

# 허깅페이스 허브에 업로드
print(f"허깅페이스 허브({hf_repo})에 업로드 시도...")
hf_dataset.push_to_hub(hf_repo, token=hf_token)
print("허깅페이스 업로드 완료!")
