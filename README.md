# Class2026_LLM

2026 LLM 수업 실습 과제 저장소.

- `hw1` ~ `hw6`: 순서대로 진행하는 과제. 각 폴더의 `README.md` 참고.
- 각 과제는 `model.py`(또는 해당 파일)의 TODO를 채우고, `test_*.py`로 확인합니다.
- **반드시 각 `hw` 폴더 안으로 들어가서 실행하세요.** (예: `cd hw3` 후 `python -m pytest test_hw3.py -v -s`)
  저장소 맨 위에서 `pytest`를 한 번에 돌리면 폴더끼리 파일 이름(`model.py` 등)이 겹쳐 오류가 납니다.
- 필요한 패키지: `pip install torch tiktoken pytest numpy`
- `solutions/`: 제출 기한이 지난 과제의 정답 예시.
