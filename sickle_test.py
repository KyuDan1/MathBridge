from sickle import Sickle
from sickle.models import Record
import requests
import os
import tarfile

# arXiv OAI 인터페이스 URL
base_url = 'http://export.arxiv.org/oai2'

# Sickle 객체 생성
sickle = Sickle(base_url)

# arXiv에서 수학 카테고리 논문 검색
records = sickle.ListRecords(metadataPrefix='arXiv', set='math')

# 논문 다운로드 폴더 설정
download_folder = './arxiv_papers_tex/'

# 다운로드 폴더가 존재하지 않으면 생성
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# 논문 TeX 소스 파일 다운로드 함수
def download_tex(record):
    identifier = record.header.identifier
    arxiv_id = identifier.split(':')[-1]
    tar_url = f'https://arxiv.org/e-print/{arxiv_id}'
    response = requests.get(tar_url, stream=True)
    if response.status_code == 200:
        tar_path = f'{download_folder}{arxiv_id}.tar.gz'
        with open(tar_path, 'wb') as f:
            f.write(response.raw.read())
        try:
            # tar 파일을 해제하여 TeX 소스 파일만 저장
            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.tex') or member.name.endswith('.TEX'):
                        member.name = os.path.basename(member.name)  # 보안 상의 이유로 경로 제거
                        tar.extract(member, path=f'{download_folder}{arxiv_id}')
            os.remove(tar_path)
            print(f'Downloaded and extracted {arxiv_id}')
        except tarfile.ReadError:
            print(f'Failed to extract {arxiv_id}: invalid tar file')
            os.remove(tar_path)
    else:
        print(f'Failed to download {arxiv_id}')

# 수학 논문 TeX 소스 파일 다운로드
for i, record in enumerate(records):
    if isinstance(record, Record):
        download_tex(record)
    if i >= 10000:  # 테스트를 위해 10개 논문만 다운로드
        break
