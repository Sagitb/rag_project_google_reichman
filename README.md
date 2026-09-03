# jiRAG

jiRAG is a Retrieval-Augmented Generation system built over an approved dataset of 1,000 anonymized Jira tickets. The project combines semantic ticket retrieval, a persistent vector store, grounded answer generation and ticket-level citations.

## Running the project

The main entry point is:

```text
jiRAG.ipynb
```

Open the notebook in Google Colab and execute its sections in order.

On the first run, Section 0:

1. Mounts Google Drive.
2. Clones the complete GitHub repository.
3. Loads the source code, requirements and approved raw dataset from the repository.
4. Creates a separate Drive directory for generated artifacts.
5. Validates the environment before data processing begins.

On later runs, the notebook updates the existing repository clone using Git and reuses compatible artifacts when possible.

## Storage structure

The GitHub repository contains the reproducible project sources:

* The main notebook.
* Python modules under `src/`.
* Project requirements and configuration.
* The approved raw dataset under `data/raw/`.

Generated files are not stored in Git. Processed datasets, splits, evaluation files, embeddings, vector indexes, reports and model artifacts are created by the notebook and stored under:

```text
MyDrive/jiRAG/artifacts/
```

The only CSV tracked in Git is the approved raw dataset. All derived CSV files are generated during execution.
