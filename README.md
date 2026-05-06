# FairTeam AI

**FairTeam AI**는 팀 프로젝트 진행 중 발생하는 무임승차, 기여도 조작, 역할 불균형, 갈등 위험을 조기에 감지하고, 교수자가 검토할 수 있는 공정평가 근거 리포트를 생성하는 AI 에이전트형 대시보드입니다.

핵심은 팀원이 직접 퍼센티지를 입력하는 방식이 아니라, 다음 원자료를 통합해 **품질 보정 기여도, 조작 의심 신호, 회의록 구조화 증거, 교수자용 리포트**를 자동 생성한다는 점입니다.

- 회의록 TXT/MD
- GitHub 커밋/PR/이슈 로그 CSV 또는 GitHub API 수집
- Google Docs 또는 보고서 수정 기록 CSV
- 발표자료 수정 기록 CSV
- 역할분담표 및 업무 완료 기록 CSV
- 팀원 자기평가 CSV

## 대상급 보강 포인트

1. **AI 회의록 구조화**  
   회의록을 팀원별 JSON형 증거로 구조화합니다. OpenAI API 키가 있으면 LLM 분석을 시도하고, 키가 없으면 동일 스키마의 규칙 기반 fallback으로 데모가 깨지지 않게 동작합니다.

2. **GitHub API 수집**  
   `github_log.csv` 업로드뿐 아니라 `owner/repo` 또는 GitHub URL을 입력해 커밋 로그를 자동 수집할 수 있습니다.

3. **점수 산식 근거 명시**  
   라인 수와 단어 수는 `sqrt` 압축을 적용해 대량 붙여넣기 과대평가를 줄이고, PR/이슈/리뷰/테스트 커밋은 검증 가능한 협업 품질 신호로 반영합니다.

4. **조작 방어 강화**  
   커밋 쪼개기, 대량 붙여넣기, 반복 커밋 메시지, 동일 파일 반복 수정, 자동 생성 파일, 문서 검증 흔적 부족, 자기평가 과장, 회의록 부정 신호를 감사합니다.

5. **교수자 다팀 비교**  
   여러 팀 폴더를 한 번에 분석해 Gini, 갈등 위험, 고위험 팀원 수, 평균 신뢰도를 비교할 수 있습니다.

## 설치

```bash
pip install -r requirements.txt
```

선택 사항: OpenAI API 기반 회의록 구조화를 쓰려면 `openai` 패키지와 환경변수가 필요합니다. 없어도 fallback으로 정상 실행됩니다.

```bash
pip install openai
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4o-mini"
```

## 대시보드 실행

```bash
streamlit run app.py
```

실행 후 사이드바의 **샘플 데이터로 데모 실행**을 켜면 바로 결과를 볼 수 있습니다.

## CLI 데모 실행

```bash
python run_demo.py
```

출력 파일:

```text
outputs/fairteam_member_scores.csv
outputs/fairteam_quality_audit.csv
outputs/fairteam_meeting_insights.csv
outputs/professor_report.md
outputs/team_report.md
outputs/scoring_policy.md
```

## GitHub API로 github_log.csv 생성

```bash
python scripts/ingest_github.py \
  --repo owner/repo \
  --branch main \
  --max-commits 120 \
  --out sample_data/github_log.csv
```

비공개 저장소나 rate limit 회피가 필요하면 token을 추가합니다.

```bash
python scripts/ingest_github.py \
  --repo https://github.com/owner/repo \
  --token "$GITHUB_TOKEN" \
  --out sample_data/github_log.csv
```

## 다팀 비교 폴더 구조

사이드바의 **팀별 입력 폴더 루트**에 아래 구조의 상위 폴더를 입력하면 교수자용 다팀 비교 탭에서 여러 팀을 비교합니다.

```text
class_teams/
├── team01/
│   ├── meeting_notes.txt
│   ├── github_log.csv
│   ├── docs_revision.csv
│   ├── slides_revision.csv
│   ├── roles.csv
│   └── self_evaluation.csv
└── team02/
    ├── meeting_notes.txt
    ├── github_log.csv
    ├── docs_revision.csv
    ├── slides_revision.csv
    ├── roles.csv
    └── self_evaluation.csv
```

## CSV 입력 형식

### github_log.csv

필수 컬럼:

```csv
member,commits,additions,deletions,files_changed,issues_closed,prs_merged,reviews,bugfix_commits,test_commits
A,28,3920,1110,46,9,12,8,7,6
```

선택 컬럼이 있으면 조작 방어가 더 강해집니다.

```csv
commit_messages,unique_message_ratio,dominant_file_ratio,generated_files,timestamp
"fix | fix | update",0.33,0.72,4,2026-05-06 22:10:00
```

### docs_revision.csv

```csv
member,edits,words_added,comments_resolved,sections_owned,suggestions_accepted,references_added
B,31,4820,18,5,14,8
```

### slides_revision.csv

```csv
member,slides_edited,visuals_created,script_words,presenter_minutes
B,16,7,980,7
```

### roles.csv

```csv
member,assigned_tasks,completed_tasks,late_tasks,critical_tasks
C,5,1,3,0
```

### self_evaluation.csv

```csv
member,self_claim_percent,claimed_main_work,peer_comment
C,25,Research and report support,I contributed ideas offline.
```

## 산출 지표

FairTeam AI는 팀원별로 다음 신호를 계산합니다.

- 코드 기여도: 커밋 수, 변경 라인 수, 파일 수, 이슈 해결, PR 병합, 리뷰, 테스트/버그픽스 커밋
- 문서 기여도: 편집 수, 추가 단어 수, 해결 댓글, 담당 섹션, 제안 반영, 참고문헌 추가
- 발표 기여도: 수정 슬라이드 수, 시각자료, 대본 작성량, 발표 담당 시간
- 회의 기여도: 참석, 발언, 액션아이템 배정 및 완료, 의사결정 언급, 회의록 구조화 신호
- 역할 이행도: 배정 업무, 완료 업무, 지연 업무, 핵심 업무
- 자기평가 불일치: 자기 주장 기여율과 로그 기반 추정치의 차이
- 품질/조작 감사: 증거 출처, 반복 패턴, 대량 변경, 마감 직전 집중, 단일 출처 의존

## 설계 원칙

이 시스템은 팀원의 최종 점수를 자동으로 확정하지 않습니다. 자동 산출값은 다음 목적의 **근거 보조 자료**입니다.

1. 교수자가 실제 기여 로그를 빠르게 검토
2. 팀원이 누락된 오프라인 기여 증거를 추가 제출
3. 팀 프로젝트 중간 단계에서 역할 불균형과 갈등을 조기에 완화
4. 조작 의심 신호가 있을 때 처벌이 아니라 원자료 재검토 지점을 표시

따라서 최종 평가는 반드시 교수자 또는 평가자가 원자료와 함께 검토해야 합니다.

## 추천 시연 흐름

1. `streamlit run app.py`
2. 샘플 데이터 결과 확인
3. **AI 회의록 구조화** 탭에서 C의 미완료/연락 지연, A의 과중 신호 확인
4. **품질/조작 감사** 탭에서 자기평가 과장, 단일 출처 의존, 반복 부정 신호 확인
5. **교수자 리포트**에서 산식 근거와 구조화 증거가 함께 들어간 Markdown 다운로드
6. 사이드바 가중치를 `report` 또는 `presentation`으로 바꿔 프로젝트 유형별 결과 변화를 시연
7. 선택적으로 `scripts/ingest_github.py`로 실제 GitHub 저장소 로그를 수집해 재실행
