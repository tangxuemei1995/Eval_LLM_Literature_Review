# Large Language Models for Automated Literature Review  
**An Evaluation of Reference Generation, Abstract Writing, and Review Composition**

## Overview
This project evaluates the performance of large language models (LLMs) on three key tasks related to automated literature reviews:

1. **Reference Generation**  
2. **Abstract Writing**  
3. **Review Composition**

## Data Access
The dataset is currently hosted on Google Drive. The download link will be made publicly available upon paper acceptance.

## Instructions

### 🔧 Setup

1. After downloading and extracting the dataset, place it in the `./clean` directory.

2. pip install -r requirements.txt

3. Use `llm_generated.py` to collect outputs from 5 LLMs across the 3 tasks.  
   Alternatively, you can directly access pre-collected results in the folders: `./t1`, `./t2`, and `./t3`.

---

### 📚 Task 1 & Task 3: Reference Generation Evaluation

**Step 1: Retrieve Candidate References**

- **Task 1 (Precision)**  
```

python evaluate\_t1.py

```

- **Task 3 (Precision)**  
```

python evaluate\_t3\_refer.py

```

- **Task 1 & Task 3 (Recall)**  
```

python evaluate\_internal\_t1\_t3.py

```

**Step 2: Compute Evaluation Metrics**

- **Precision (Task 1 & 3)**  
```

python metrics\_t1\_t3.py

```

- **Recall (Task 1 & 3)**  
```

python metrics\_t1\_t3\_internal.py

```

---

### 📝 Task 2: Abstract Writing Evaluation

- **Semantic Similarity**  
```

python metric\_similarity.py

```

- **NLI Evaluation (TRUE model)**  
```

python metrics\_t2\_true.py

```

- **NLI Evaluation (GPT-4o model)**  
```

python metrics\_nli.py

```

- **ROUGE Score**  
```

python metric\_rouge.py

```

---

### 🧠 Task 3: Review Composition Evaluation

- **ROUGE Score**  
```

python metric\_t3\_rouge.py

```

- **Key Point Extraction**  
```

python extract\_point.py

```

- **Key Point Recall Evaluation**  
```

python evaluate\_t3\_kpr.py

```

---

## Citation  
(To be added upon paper acceptance.)
```

---


