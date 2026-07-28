import os
from datetime import datetime

# 路径配置（BIMS_DATA_ROOT 用于并行评测隔离，避免双卡抢同一 talk/vectors）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("BIMS_DATA_ROOT", os.path.join(BASE_DIR, "data"))
TALK_FILE = os.path.join(_DATA_ROOT, "talk.txt")
VECTOR_DB_DIR = os.path.join(_DATA_ROOT, "vectors")
KNOWLEDGE_DIR = os.path.join(_DATA_ROOT, "knowledge")

# 创建目录
os.makedirs(os.path.dirname(TALK_FILE), exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# Ollama配置（评测可通过环境变量 OLLAMA_BASE_URL 指向 GPU 实例，如 http://127.0.0.1:11435）
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
GENERATION_MODEL = "qwen3:14b"

# GPU配置（CUDA_VISIBLE_DEVICES 在评测启动脚本中按此设置）
CUDA_DEVICE = os.environ.get("CUDA_DEVICE", "1")
GPU_LAYERS = -1    # -1表示使用所有可用层，或指定具体层数如 20

# 向量配置
VECTOR_DIM = 768
TOP_K_RETRIEVAL = 3
# 嵌入 API 单次最大字符数（法律长答案需截断，避免 Ollama 超时/失败）
EMBED_MAX_CHARS = int(os.environ.get("EMBED_MAX_CHARS", "500"))
EMBED_MAX_RETRIES = int(os.environ.get("EMBED_MAX_RETRIES", "3"))
# 批量嵌入：Ollama /api/embed 一次请求多条 input（建库阶段显著加速）
USE_EMBED_BATCH = os.environ.get("USE_EMBED_BATCH", "0").lower() in ("1", "true", "yes")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "384"))
# 嵌入后端：ollama（HTTP）| gpu_local（进程内 SentenceTransformer，高 GPU 利用率）
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "ollama").lower()
EMBED_OLLAMA_WORKERS = int(os.environ.get("EMBED_OLLAMA_WORKERS", "24"))
EMBED_HF_MODEL = os.environ.get("EMBED_HF_MODEL", "nomic-ai/nomic-embed-text-v2-moe")
EMBED_GPU_BATCH_SIZE = int(os.environ.get("EMBED_GPU_BATCH_SIZE", "768"))
EMBED_GPU_FP16 = os.environ.get("EMBED_GPU_FP16", "1").lower() in ("1", "true", "yes")
EMBED_DISK_CACHE_DIR = os.environ.get(
    "EMBED_DISK_CACHE_DIR",
    os.path.join(_DATA_ROOT, "embed_cache", "vectors.sqlite"),
)
USE_EMBED_DISK_CACHE = os.environ.get("USE_EMBED_DISK_CACHE", "0").lower() in ("1", "true", "yes")

# 记忆配置
MAX_DIALOG_HISTORY = 50
CACHE_SIZE = 1000
CLUSTER_UPDATE_THRESHOLD = 5

# 推理配置
INFERENCE_TIMEOUT = 120  # 增加超时时间以适应GPU推理
