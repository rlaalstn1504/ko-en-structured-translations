# Korean–English Multistyle Parallel Corpus  
한국어 사용자에게 익숙한 표현 기반의 다도메인·다문체 한–영 병렬 코퍼스


## 소개(Introduction)

저는 머신러닝, 인공지능 수업을 진행하는 강사입니다.  
Seq2Seq, Attention, Transformer 등 자연어처리(NLP) 수업을 진행하며  
한국 학습자에게 **자연스럽고 익숙한 한–영 번역 데이터셋의 부족**을 경험했습니다.  

기존 공개 데이터셋은  
- 도메인 다양성이 부족하거나  
- 문체가 한국 사용자에게 자연스럽지 않거나  
- 전반적으로 문장의 퀄리티가 매우 부족하여

**학습한 번역 모델의 실제 성능이 기대만큼 나오지 않는 문제**가 있었습니다.

이 문제를 해결하기 위해, 딥러닝 강사로서 langchain을 사용하여 **직접 고품질 병렬 데이터를 자동으로 생성·정제하여 구성한 데이터셋**입니다.  

한국어 사용자에게 익숙한 표현을 중심으로 다양한 문체, 문장 구조를 포함해  
교육용·연구용·실습용으로 바로 사용 가능한 실용적 번역 코퍼스를 목표로 합니다.



## 데이터셋 개요(Dataset Overview)

| 항목 | 내용 |
|------|------|
| **총 샘플 수** | 9,493개 |
| **train** | 7,594 |
| **valid** | 949 |
| **test** | 950 |
| **언어쌍** | 한국어(ko) – 영어(en) |
| **도메인(topic)** | 20개 (여행, 일상, 건강, 식당, IT/AI, 스포츠, 교육, 감정표현 등) |
| **문체(style)** | 8개 (대화체, SNS, 보고서체, 기술 설명체, 감정표현, 고객센터 등) |
| **문장 유형(type)** | 6개 (단문, 복문, 조건문, 비교문, 원인·결과문, 명령/요청문) |
| **생성 모델** | gpt-4o-mini |
| **프롬프트 버전** | v2_structured |
| **라이선스** | MIT License |

---

### 파일 구성(File Structure)

```
dataset/
 ├── train.csv
 ├── valid.csv
 ├── test.csv
 └── parallel_data_gpt-4o-mini_v2_structured_YYYYMMDD_HHMMSS.csv(전체파일, 버전관리용 이름)
```

CSV 컬럼:

| 컬럼명 | 설명 |
|--------|-------|
| `id` | 고유 ID |
| `topic` | 문장의 주제 |
| `style` | 문체 스타일 |
| `type` | 문장 구조 |
| `ko` | 한국어 문장 |
| `en` | 영어 문장 |
| `model` | 생성 모델명 |
| `prompt_version` | 프롬프트 버전 |
| `created_at` | 생성 timestamp |
| `split` | train/valid/test 구분 |

---

## 데이터 예시(Examples)

| topic | style | type | ko | en |
|-------|--------|--------|-----|-----|
| 건강 | 대화체 | 조건문 | 스트레스를 관리하지 않으면 문제가 생길 수 있어. | If you don't manage your stress, problems may arise. |
| 경제/금융 | 감정 표현 스타일 | 비교문 | 이 주식의 수익률은 다른 주식들보다 더 높아. | The return on this stock is higher than that of other stocks. |
| 스포츠 | 기술 설명체 | 조건문 | 체스에서는 전략이 승패를 좌우하니까 깊게 생각해야 해. | In chess, strategy determines the outcome, so you need to think deeply. |
| 고객센터 | 고객센터 안내 스타일 | 단문 | 신속한 처리 부탁드려요. | I would appreciate a quick resolution. |
| 식당 | SNS 스타일 | 명령/요청문 | 테이크아웃 가능해요? | Is takeout available? |

---

## 데이터 생성 방식(How It Was Created)

이 데이터셋은 **LangChain + OpenAI GPT-4o-mini**를 이용해  
topic × style × type 조합별로 구조화된 JSON 형태의 병렬 문장을 생성한 뒤,  
길이/중복/문법 기반 필터링을 거쳐 구축되었습니다.

생성 조건 요약:

- 한국어 10–40자, 영어 10–80자 범위  
- 문체(반말/해요체/격식체)를 균형 있게 포함  
- JSON 구조 준수 (pairs → ko/en)  
- 의미 대응 정확성 유지  
- 직역투 지양  
- 중복 제거 후 무작위 셔플  
- train/valid/test = 8:1:1 자동 분할

---

## 생성 코드(Generation Pipeline)

make_dataset.ipynb의 코드를 실행하면 누구나 같은 구조로 데이터를 확장 생성할 수 있습니다. 
.env 파일을 만들어 해당 파일에 OPENAI_API_KEY를 넣어야 정상적으로 동작합니다. 


## 사용 방법(Usage)

```python
from datasets import load_dataset

ds = load_dataset("username/ko-en-multistyle-corpus")

print(ds["train"][0])
```

---

## License

이 데이터셋은 **MIT License** 하에 자유롭게 사용할 수 있습니다.  
사용자는 출처를 유지하는 조건으로 상업적/비상업적 목적 모두 활용 가능합니다.

---

### 만든 이유와 의의

이 데이터셋은 **처음 머신러닝과 딥러닝을 배우는 학생들의 번역 모델**을  
보다 효과적으로 학습시키기 위해 제작되었습니다.

특히 다음 용도에 적합합니다:

- Transformer / Seq2Seq 실습  
- 번역 모델 튜닝  
- 학습자 프로젝트용 번역 데이터 구축  
- LLM 기반 번역 품질 향상 실험

실제 강의 환경에서 모델 성능이 기존 공개 데이터셋 대비 더 자연스럽고 정확한 출력으로 개선됨을 확인했습니다.

---

## Contact

김민수 강사  
📧 rlaalstn1504@naver.com

교육·연구·협업 문의 언제든 환영합니다.
