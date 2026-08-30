## Quick Start

The project is designed to run in Google Colab using the main `jiRAG.ipynb` notebook.

### First run

1. Open `jiRAG.ipynb` in Google Colab.

2. Select **Runtime → Run all**.

3. Authorize Google Drive when prompted.

4. If the raw dataset is not already available, the notebook will request the following file:

   ```text
   jira_rag_master_FINAL_1000.csv
   ```

   The file is included in this repository under:

   ```text
   data/raw/jira_rag_master_FINAL_1000.csv
   ```

5. Upload that exact file when the upload widget appears.

The notebook saves the raw dataset persistently to:

```text
MyDrive/jiRAG/data/raw/jira_rag_master_FINAL_1000.csv
```

It then automatically creates and validates all derived artifacts, including:

* the normalized dataset;
* QA and EDA reports;
* Train, Validation and Test splits;
* manifests and fingerprints;
* RAG document JSONL files.

### Subsequent runs

If the raw dataset and generated artifacts already exist in Google Drive, the notebook reuses and validates them. The upload widget will not appear again unless the raw dataset is missing.

Generated datasets, splits, reports, indexes, checkpoints and models are stored in Google Drive and are not committed to Git.
