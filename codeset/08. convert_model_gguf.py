try:
    import google.colab
    inColab = True
except ImportError:
    inColab = False
if inColab == True:
    from google.colab import drive
    drive.mount("/content/gdrive")

### ★★★ 수정 포인트 ★★★ 
# 코드의 위치 로 이동
if inColab == True:
    %cd /content/gdrive/MyDrive/Colab Notebooks/ggufs

### HF TO SNAPSHOT
### ★★★ 수정 포인트 ★★★
from huggingface_hub import snapshot_download
# 현재 폴더 기준 gdownPath 폴더가 생성되고 허깅페이스의 model_id 이름의 모델이 다운로드 됨
gdownPath = "gemma3_4b_ko"
# 불러올 허깅페이스모델 (model_id 변수에 할당)
model_id="hyokwan/familidata_elysium_4b"

# 현재 폴더 기준 폴더 생성
# 구글드라이브에 이미 있으면 두번 실행할 필요는 없다!
snapshot_download(repo_id=model_id, local_dir=gdownPath,
                          local_dir_use_symlinks=False, revision="main")
# Setup Llama.cpp and install required packages

# 구글드라이브에 이미 있으면 두번 실행할 필요는 없다!
!git clone https://github.com/ggerganov/llama.cpp
!pip install -r llama.cpp/requirements.txt
####### ★★★ 수정 포인트★★★ ############
# !python llama.cpp/convert_hf_to_gguf.py [파인튜닝모델 다운로드 폴더위치] --outfile [GGUF 파일명].gguf --outtype {f32, f16, bf16, q8_0, auto}
!python llama.cpp/convert_hf_to_gguf.py gemma3_4b_ko --outfile familidata_gemma3_4b_elysium.gguf --outtype q8_0
from huggingface_hub import HfApi
api = HfApi()

####### ★★★ 수정 포인트★★★ ############
# gguf 파일 경로
file_path = "familidata_gemma3_4b_elysium.gguf"

# Hugging Face 저장소 이름
repo_name = "hyokwan/familidata_elysium_gguf"

# 파일 업로드
api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo="familidata_gemma3_4b_elysium.gguf",  # 저장소 내 파일 경로
    repo_id=repo_name,
    repo_type="model",  # 모델 저장소에 업로드
)