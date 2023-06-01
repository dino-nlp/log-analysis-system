# Log Analysis system

### Introduction
This is a log analysis system that is able to parse log files and extract information to detect anomalies.

### Technologies
- Machine Learning
  - BERT
  - MLFlow
- Data Engineering
  - Airbyte
  - Airflow
  - Kafka
  - dbt
  - Great Expectations

### Architecture

// TODO

### Virtual environment
```bash
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit autoupdate
```

### Directory
```bash
tagifai/
├── data.py       - data processing components
├── drain.py      - parser raw log files to structured data
├── logbert.py    - training, inference components
└── utils.py      - supplementary utilities
...
```

### Workflow
```bash
python logparser/data.py
python logparser/logbert.py vocab
```

### API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```

### References
- https://github.com/HelenGuohx/logbert
- https://github.com/GokuMohandas/mlops-course
