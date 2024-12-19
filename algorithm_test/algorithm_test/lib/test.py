import os
import sys
import pandas as pd

def read_wp_csv():
    # 실행된 스크립트의 디렉토리 경로 가져오기
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    print(f"Script directory: {script_dir}")
    # wp.csv 파일의 전체 경로 생성
    csv_path = os.path.join(script_dir, 'wp.csv')
    
    # 파일이 존재하는지 확인
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' 파일을 찾을 수 없습니다.")
    
    # wp.csv 파일 읽기
    data = pd.read_csv(csv_path)
    return data
