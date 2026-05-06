# FairTeam AI

**FairTeam AI**는 팀 프로젝트 진행 중 발생하는 무임승차, 기여도 조작, 역할 불균형, 갈등 위험을 조기에 감지하고, 교수자가 검토할 수 있는 공정평가 근거 리포트를 생성하는 AI 에이전트형 대시보드입니다.

이 프로젝트의 핵심은 사용자가 퍼센티지를 직접 입력하는 것이 아니라, 다음 원자료를 넣으면 시스템이 자동으로 팀원별 기여도 추정치와 근거를 계산한다는 점입니다.

- 회의록
- GitHub 커밋/PR/이슈 로그
- Google Docs 또는 보고서 수정 기록
- 발표자료 수정 기록
- 역할분담표 및 업무 완료 기록
- 팀원 자기평가

## 본 프로젝트의 차별화 포인트

1. **학생 체감 문제**: 무임승차, 역할 불균형, 자기평가 과장, 팀원 갈등은 거의 모든 팀플에서 발생합니다.
2. **교수자 활용성**: 최종 점수를 자동 결정하지 않고, 로그 기반 근거 리포트를 제공하여 공정평가를 보조합니다.
3. **AI 활용 명확성**: 회의록에서 담당자·마감일·갈등 문장을 추출하고, Git/문서/발표/회의 로그를 통합하여 기여도와 위험 신호를 요약합니다.
4. **데모 강도**: 샘플 데이터가 포함되어 있어 실행 즉시 기여도 대시보드와 교수자 리포트를 볼 수 있습니다.

## 기능

- 팀원별 코드/문서/발표/회의/역할 기여도 자동 산출
- 회의록에서 담당자, 액션아이템, 완료 여부, 갈등 문장 감지
- GitHub 로그 기반 코드 기여도 계산
- 문서·발표자료 수정 기록 기반 비코드 기여도 계산
- 자기평가와 실제 로그 기반 기여도 차이 감지
- 무임승차 위험, 과장 자기평가, 업무 과중 위험 태그 생성
- 교수자용 공정평가 리포트 Markdown/CSV 다운로드
- 프로젝트 유형별 가중치 조정: 개발형, 보고서형, 발표형, 균형형

## 설치

```bash
pip install -r requirements.txt
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
outputs/professor_report.md
outputs/team_report.md
```

## 샘플 데이터 구조

```text
sample_data/
├── meeting_notes.txt
├── github_log.csv
├── docs_revision.csv
├── slides_revision.csv
├── roles.csv
└── self_evaluation.csv
```

## CSV 입력 형식

### github_log.csv

```csv
member,commits,additions,deletions,files_changed,issues_closed,prs_merged,reviews,bugfix_commits,test_commits
A,28,3920,1110,46,9,12,8,7,6
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
- 회의 기여도: 참석, 발언, 액션아이템 배정 및 완료, 의사결정 언급
- 역할 이행도: 배정 업무, 완료 업무, 지연 업무, 핵심 업무
- 자기평가 불일치: 자기 주장 기여율과 로그 기반 추정치의 차이

## 설계 원칙

이 시스템은 팀원의 최종 점수를 자동으로 확정하지 않습니다. 자동 산출값은 다음 목적의 **근거 보조 자료**입니다.

1. 교수자가 실제 기여 로그를 빠르게 검토
2. 팀원이 누락된 오프라인 기여 증거를 추가 제출
3. 팀 프로젝트 중간 단계에서 역할 불균형과 갈등을 조기에 완화

따라서 최종 평가는 반드시 교수자 또는 평가자가 원자료와 함께 검토해야 합니다.

## 추천 시연 흐름

1. `streamlit run app.py`
2. 샘플 데이터 결과 확인
3. C 팀원의 낮은 로그 기여도와 자기평가 차이 확인
4. A의 코드 과중 위험, B의 문서/발표 중심 기여, D의 테스트/리뷰 기여 확인
5. 교수자 리포트 다운로드
6. 사이드바 가중치를 `report` 또는 `presentation`으로 바꿔 프로젝트 유형별 결과 변화를 시연

## 확장 아이디어

- GitHub API, Google Drive API 연동
- 학교 LMS 팀플 제출물 이력 연동
- 팀원별 이의제기 워크플로우
- 교수자 대시보드에서 여러 팀 일괄 비교
- 오프라인 기여 증빙 업로드 및 반영
- LLM 기반 회의록 요약/중재 문안 생성
