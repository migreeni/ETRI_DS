# automatic experiment
# 각 명령어 개별 입력 혹은 sh파일 실핼 -> 터미널에 'run.sh' 입력
# sh파일 실행시 리눅스(맥)은 아무거나, 윈도우는 반드시 git bash에 입력할 것

# 1. preprocessed
# python preprocess_original_final.py
# python preprocess_dwt_final.py

# 2. prediction for original data
python main.py -d 'original' -m 'rf'
python main.py -d 'original' -m 'lgbm'
python main.py -d 'original' -m 'xgb'
python main.py -d 'original' -m 'cat'
python main.py -d 'original' -m 'lr'
python main.py -d 'original' -m 'ensemble'
python main.py -d 'original' -m 'stacking'

# 2. prediction for dwt data
python main.py -d 'dwt' -m 'rf'
python main.py -d 'dwt' -m 'lgbm'
python main.py -d 'dwt' -m 'xgb'
python main.py -d 'dwt' -m 'cat'
python main.py -d 'dwt' -m 'lr'
python main.py -d 'dwt' -m 'ensemble'
python main.py -d 'dwt' -m 'stacking'