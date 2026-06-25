# 🔬 SK하이닉스 조직·업무 AI Agent

자연어로 SK하이닉스 조직 R&R 정보를 검색하는 AI Agent (데모용 가상 데이터)

## 📁 파일 구조

```
sk_hynix_agent/
├── app.py                              # Streamlit 메인 앱
├── requirements.txt                    # Python 의존성
├── sk_hynix_dummy_org_rnr_300.csv     # 가상 직원 데이터 (300명)
└── README.md
```

## 🏗️ Agent 아키텍처

```
        사용자 자연어 질문
               │
        AI Agent (GPT)
               │
  ┌────────────┴────────────┐
  │                         │
Vector DB               CSV Database
(TF-IDF 벡터 검색)      (키워드/조직도 검색)
  │                         │
  └────────────┬────────────┘
        결과 통합 & 컨텍스트 구성
               │
        GPT 최종 답변 생성
               │
       Web UI (Chat 형태)
```

- **Vector DB**: TF-IDF + Cosine Similarity, char 2~4gram 기반 의미 검색
- **CSV DB**: pandas 직접 검색, 이름/팀명 정확 매칭 보완
- **하이브리드**: 두 검색 결과를 통합하여 LLM에 컨텍스트로 제공

## 🚀 배포 방법

### 방법 1: Streamlit Community Cloud (무료, URL 공유 가능) ✅ 권장

1. GitHub에 이 폴더 내용 업로드
2. [share.streamlit.io](https://share.streamlit.io) 접속
3. GitHub 저장소 연결 후 `app.py` 선택
4. Deploy → 공유 가능한 URL 생성됨
5. 접속한 사용자가 각자 OpenAI API 키 입력

### 방법 2: 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501 접속
```

### 방법 3: Railway / Render 배포

- Dockerfile 없이 `requirements.txt` + `app.py`만으로 배포 가능
- 시작 명령어: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## 🔑 사용 방법

1. 사이드바에 OpenAI API Key 입력 (`sk-...`)
2. 자연어로 질문 입력
   - "DRAM 수율 담당자는 누구인가요?"
   - "품질 검사 관련 업무는 어느 팀이 맡나요?"
   - "노광 공정 담당자 알려줘"
3. AI가 Vector DB + CSV DB 검색 후 GPT로 자연스러운 답변 생성

## ⚠️ 주의사항

- 본 시스템은 **데모용 가상 데이터**를 사용합니다
- 실제 SK하이닉스 직원 정보와 무관합니다
- API Key는 서버에 저장되지 않으며 브라우저 세션에서만 사용됩니다
