# Hybrid Book Recommendation System

## Project Overview

This project implements a complete, end-to-end personalized book recommendation system based on the Book-Crossing dataset. The primary goal was to build a hybrid model that leverages both collaborative filtering and content-based features to provide relevant recommendations for users. The entire pipeline, from advanced data cleaning and feature engineering to model training, hyperparameter optimization, and deployment via an interactive dashboard, is production-ready.

**Core Task:** To build a system that can effectively learn user and item representations to provide:
1.  Personalized recommendations for known users.
2.  Similar item recommendations based on content.
3.  Robust fallback strategies for new ("cold start") users and items.

---

## Key Features & Technical Decisions

This project is not just a simple model; it's a demonstration of a  ML system design. Key technical features include:

*   **Advanced Data Cleaning & Deduplication:**
    *   **Problem:** The raw data contained significant noise, including character encoding errors, inconsistent author names, and multiple ISBNs for the same conceptual book.
    *   **Solution:** Implemented a multi-stage cleaning pipeline that:
        1.  Repairs Unicode text errors (`ÃƒÂ©` -> `é`).
        2.  Applies advanced normalization to author and title fields, including robustly removing edition-specific text.
        3.  Uses a **string similarity (`Jaro-Winkler`)** and **graph-based clustering (`networkx`)** approach to intelligently group different editions of the same book under a single `work_id`. This is a lossless enrichment process.

*   **Hybrid Two-Tower Model Architecture:**
    *   **Why:** This SOTA architecture is highly scalable and effectively combines multiple feature types.
    *   **Implementation:**
        *   **User Tower:** Learns representations from `user_id`, `age_bin`, and `country`.
        *   **Item Tower:** Learns representations from `work_id`, `canonical_author`, `publisher`, `yop_bin`, and, crucially, **pre-computed text embeddings** from a Hugging Face `SentenceTransformer` model (`all-MiniLM-L6-v2`) for book titles.

*   **Ranking-Focused Objective:**
    *   **Problem:** Initial regression-based training (`MSELoss`) resulted in poor ranking performance (low NDCG).
    *   **Solution:** Pivoted to a more appropriate, SOTA-informed **ranking objective** using `BCEWithLogitsLoss`. This directly trains the model to distinguish between "liked" and "not liked" items, which is perfectly aligned with the goal of creating a ranked recommendation list.

*   **Automated Hyperparameter Optimization:**
    *   **Why:** To move beyond guesswork and find a demonstrably better model configuration.
    *   **Tool:** Used the **Optuna** framework to perform an automated search over key hyperparameters like embedding dimensions, learning rate, dropout, and MLP depth. The final model was trained using the champion parameters found in this study.

*   **Scalable Hybrid Inference with Vector Databases:**
    *   **Why:** To provide real-time recommendations, even with a large catalog.
    *   **Architecture:**
        1.  An **Offline Collaborative Index (`faiss_index.bin`):** Built from the trained `ItemTower` embeddings. Used for fast user-to-item recommendations.
        2.  An **Offline Semantic Index (`semantic_faiss_index.bin`):** Built directly from the Hugging Face title embeddings. Used for intuitive, content-based "similar item" recommendations.
    *   This two-index approach ensures that each feature provides the most logical and high-quality user experience.

---

## Project Structure
.
├── .env # Configuration for data paths
├── README.md # This file
├── requirements.txt # All Python dependencies
├── dashboard.py # The final streamlit interactive dashboard
└── full_pipeline.ipynb # Notebook with the complete end-to-end workflow
|
├── data/
│ ├── # Original dataset CSVs
│ └── processed/ # All generated assets for the dashboard│

---

## How to Run

**1. Setup:**

*   Clone the repository: `git clone ...`
*   Create and activate a Python virtual environment (e.g., `python3 -m venv venv && source venv/bin/activate`).
*   Install all dependencies: `pip install -r requirements.txt`
*   Create a `.env` file in the root directory with the content: `PROCESSED_DATA_DIR="data/processed"`
*   Place the raw dataset files (`Books.csv`, etc.) into the `data/raw/` directory.

**2. Run the Data & Modeling Pipeline:**

*   Open and run all cells in the `notebooks/full_pipeline.ipynb` notebook from top to bottom.
*   This will perform all data cleaning, model training, HPO, and will generate all the necessary assets (`.pth` model, `.json` config, `.bin` indexes, etc.).

**3. Launch the Interactive Dashboard:**

*   From your terminal (with the virtual environment activated), run the following command:
    ```bash
    streamlit run dashboard.py
    ```
*   Open the local URL provided  in your web browser.

---

## Final Model Performance

The final champion model was trained for 30 epochs using the optimal hyperparameters found by Optuna. The performance was evaluated on a held-out validation set of interactions from our core user group.

| Metric         | Value  | Interpretation                                                                                                  |
| :------------- | :----- | :-------------------------------------------------------------------------------------------------------------- |
| **NDCG@10**    | 0.9347 | **Excellent.** The model is extremely effective at placing the most relevant items at the top of the recommendation list. |
| **Precision@10**| 0.2333 | **Strong.** On average, over 2 of the 10 recommended books are highly relevant. The metric appears modest because the validation set for each user contains very few relevant items (often only 1-2 "correct answers"). Achieving this level of precision on such a sparse task is a strong indicator of effective personalization.                      |
| **Recall@10**  | 0.9829 | **Very High.** Indicates the model is successful at finding the few relevant items present in the small validation sets for each user. |

---

## Cold Start Strategy

A key part of this project was designing a robust system for handling new users and items. Our final system employs a **tiered fallback strategy**:

*   **Cold Start Users:** Are shown a randomized list of the Top 20 most popular books on the platform. This provides a high-quality, diverse, and non-repetitive experience.
*   **Cold Start Items (for "Similar Items" feature):** If a user selects a book the model wasn't trained on, the system falls back to recommending other books by the **same author**. If no other books by that author exist, it falls back again to showing the most popular books. This ensures the user never sees an empty recommendation list.

---

## Future Improvements & Architectural Considerations

While this project implements a robust, end-to-end system, a production-grade recommender is an ever-evolving product. Based on our analysis during development, here are the key, high-impact improvements that would be prioritized next:

1.  **Curated Author Alias Map (Gazetteer):**
    *   **Limitation:** Our current author normalization is powerful but can't resolve ambiguous cases like `"Shakespeare"` vs. `"William Shakespeare"`. This results in some authors being treated as separate entities, slightly fracturing our item clusters.
    *   **Solution:** The next step would be to implement a curated alias map (a simple dictionary or a database table) that maps known variations of an author's name to a single, canonical entity. This is a low-risk, high-impact fix that would perfect our entity resolution or use lm model to correct this type issue.

2.  **Incorporate User Interaction Weights:**
    *   **Opportunity:** We successfully engineered a `rating_count` feature, which identifies "super-fans" who have rated multiple editions of the same book.
    *   **Next Step:** Incorporate this `rating_count` as a **sample weight** in the training loop to force the model to pay more attention to the preferences of our most passionate users, potentially leading to even better personalization.

3.  **Advanced Negative Sampling:**
    *   **Limitation:** Our current BCE loss treats all non-interacted items with a low rating as simple "negative" examples.
    *   **Solution:** Implement a more advanced sampling strategy, such as "in-batch negative sampling" or sampling hard negatives, to provide the model with more informative training signals.

4.  **Temporal Feature Engineering:**
    *   **Opportunity:** We currently bin the `yop` (year of publication) into static eras.
    *   **Next Step:** A more dynamic approach would be to treat the age of a book (relative to the present day) as a feature or to incorporate rating timestamps (if available) to model evolving user tastes over time using sequential recommendation models.

5.  **Robust A/B Testing Framework:**
    *   **Final Step to Production:** While our offline metrics are strong, a production system would require an A/B testing framework to test our model against baselines and measure its true impact on key business metrics like click-through rate, conversion, and user retention.
6.  **Transitioning to Advanced Architectures (LLMs/RAG):** The Two-Tower model is a highly efficient SOTA baseline. To unlock the power of next-generation models like Large Language Models with Retrieval-Augmented Generation (RAG), the primary prerequisite is richer data. Our current dataset lacks critical metadata such as genre, abstracts, user-written reviews, and items_bought_together signals. An essential next step would be to build data ingestion pipelines to enrich our item catalog with this information. With a richer feature set, we could then effectively leverage fine-tuned LLMs (using SFT or LoRA) and advanced vector databases (Qdrant, etc.) to provide recommendations with deep semantic understanding and conversational capabilities.

# Code, Data, and Model Versioning

While a full production-grade implementation of DVC (Data Version Control) is not included in this assignment, the following outlines a SOTA approach for versioning data, features, and models. This design draws from modern MLOps best practices as of 2025, emphasizing reproducibility, scalability, and integration with distributed systems. It incorporates tools like Kafka for real-time ingestion, NiFi for data flows, dbt for transformations, Apache Airflow for orchestration, Spark for large-scale processing, MinIO for object storage, Qdrant for vector embeddings, Neo4j for graph-based relationships, Kubernetes (K8s) for deployment, and ClickHouse for analytical queries on metrics/logs.

## Data and Feature Versioning Approach
- **Core Strategy**: Treat data and features as code—use Git-integrated tools like DVC to track versions without bloating the repo. Raw data (e.g., CSVs) and processed features (e.g., embeddings, canonical mappings) are versioned via pointers (`.dvc` files) that reference external storage. This ensures immutable snapshots for reproducibility.
  - **Ingestion**: Use Kafka or NiFi for real-time ELT pipelines to stream user interactions/ratings into a landing zone, handling schema evolution automatically.
  - **Processing**: Leverage Spark for distributed cleaning/deduplication (e.g., graph clustering on 271K books) and dbt for SQL-based transformations (e.g., feature bins like `age_bin`).
  - **Storage**: Store versioned artifacts in MinIO (S3-compatible object store) for cheap, scalable persistence—e.g., `dvc add data/processed && dvc push` uploads Parquet files/embeddings to MinIO buckets.
  - **Graph/Vector Enhancements**: For SOTA relational features (e.g., author graphs), version Neo4j snapshots (export Cypher dumps). Embeddings are versioned in Qdrant (vector DB) for fast similarity queries, replacing Faiss with hybrid dense/sparse indexing for better cold-start handling.

## Model Versioning Approach
- **Core Strategy**: Use semantic versioning (e.g., v1.0.0) with metadata tags for models, ensuring traceability back to data versions and experiments. Models are registered post-training, with automated promotion (dev → staging → prod) based on metrics.
  - **Tracking**: Integrate MLflow (or similar) to log params (from Optuna), artifacts (`final_ranking_model.pth`), and metrics (NDCG@10) during training—e.g., `mlflow.log_artifact('faiss_index.bin')`.
  - **Orchestration**: Apache Airflow DAGs illustrate the pipeline: `ingest (Kafka) → transform (dbt/Spark) → train (PyTorch) → evaluate → register (MLflow) → deploy`. A sample DAG could trigger on data changes, with retries and monitoring.
  - **Deployment**: Package models in Docker containers orchestrated by Kubernetes (K8s) for scalable serving—e.g., auto-scale pods for inference traffic.
  - **Analytics**: Log all versions/metrics to ClickHouse for fast querying (e.g., "SELECT avg(NDCG) FROM models WHERE version > 'v1.0'"), enabling A/B tests and drift detection.

This approach ensures auditability (e.g., reproduce v1.2 from specific data hash) while scaling to production e-commerce loads, aligning with SOTA MLOps from AWS/Netflix patterns. For this assignment, Git + manual MLflow simulates it; full setup would use CI/CD (GitHub Actions) for automation.
* Graph of  complete pipeline 
<div align="center">
  <img width="501" alt="dependency_dag_chart_v2" src="https://github.com/user-attachments/assets/4d0d2b9f-a600-4259-a172-f80913a7d61e">
</div>
