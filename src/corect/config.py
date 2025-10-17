DATASETS_FOLDER = "datasets"
MODELS_FOLDER = "models"
RESOURCES_FOLDER = "resources"
RESULTS_FOLDER = "results"
SHARE_RESULTS_FOLDER = "share_results"
EMBED_FOLDER = "embed"
INDEX_FOLDER = "index"
MODEL_FOLDER = "model"
DIMENSIONALITIES = [1024, 512, 256, 128, 64, 32] # Snowflake: [768, 384, 256, 192, 96, 48, 24]
CORPUS_CHUNK_SIZE = 50_000
K_VALUES = [1, 3, 5, 10, 20, 100, 200, 300, 500, 1000]
METRICS = {
    "ndcg_at_10": "NDCG@10",
    "recall_at_100": "Recall@100",
    "recall_at_1000": "Recall@1000",
}
EPOCHS = 100
