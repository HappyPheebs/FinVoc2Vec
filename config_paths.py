# 路径配置文件
# 所有脚本的路径统一在这里管理

import os

# ==================== 项目根目录 ====================
PROJECT_ROOT = r"D:\project\Friendly-Downloader"

# ==================== 模型相关路径 ====================
# FinVoc2Vec 模型目录（包含 model.safetensors, config.json 等）
MODEL_DIR = os.path.join(PROJECT_ROOT, "finvoc2vec_code")

# 模型文件
MODEL_SAFETENSORS = os.path.join(MODEL_DIR, "model.safetensors")
MODEL_BIN = os.path.join(MODEL_DIR, "pytorch_model.bin")
MODEL_CONFIG = os.path.join(MODEL_DIR, "config.json")
PREPROCESSOR_CONFIG = os.path.join(MODEL_DIR, "preprocessor_config.json")

# ==================== 数据目录 ====================
# 路演下载数据根目录
ROADSHOW_DIR = os.path.join(PROJECT_ROOT, "roadshow_downloads")

# 音频文件目录
AUDIO_DIR = os.path.join(ROADSHOW_DIR, "audio")

# 视频文件目录
VIDEO_DIR = os.path.join(ROADSHOW_DIR, "videos")

# 转录文件目录
TRANSCRIPTS_DIR = os.path.join(ROADSHOW_DIR, "transcripts")
SRT_DIR = os.path.join(TRANSCRIPTS_DIR, "srt")

# ==================== 输出目录 ====================
# 测试输出目录
TEST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_output")

# NPY向量输出目录（如果需要单独保存）
NPY_OUTPUT_DIR = os.path.join(ROADSHOW_DIR, "embeddings")

# ==================== Excel 文件路径 ====================
# 下载记录文件
DOWNLOAD_RECORD_XLSX = os.path.join(ROADSHOW_DIR, "下载记录.xlsx")
DOWNLOAD_RECORD_WITH_EMOTION_XLSX = os.path.join(ROADSHOW_DIR, "下载记录_带情感结果.xlsx")

# 语音情感分析结果文件
VOCALTONE_XLSX = os.path.join(PROJECT_ROOT, "output_Vocaltone.xlsx")
VOCALTONE_WITH_EMOTION_XLSX = os.path.join(PROJECT_ROOT, "output_Vocaltone_带情感结果.xlsx")

# 转录汇总文件
TRANSCRIPT_SUMMARY_XLSX = os.path.join(TRANSCRIPTS_DIR, "转录汇总.xlsx")

# ==================== 其他配置 ====================
# 默认采样率
DEFAULT_SAMPLE_RATE = 16000

# 默认分段时长（秒）
DEFAULT_SEGMENT_DURATION = 5.0

# 默认设备
DEFAULT_DEVICE = "cpu"

# Excel 匹配列名
EXCEL_KEY_COLUMN = "文件名"


# ==================== 辅助函数 ====================
def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)
    return path


def get_all_paths():
    """返回所有路径的字典，方便查看"""
    return {
        "项目根目录": PROJECT_ROOT,
        "模型目录": MODEL_DIR,
        "音频目录": AUDIO_DIR,
        "视频目录": VIDEO_DIR,
        "转录目录": TRANSCRIPTS_DIR,
        "SRT目录": SRT_DIR,
        "测试输出目录": TEST_OUTPUT_DIR,
        "NPY输出目录": NPY_OUTPUT_DIR,
        "下载记录": DOWNLOAD_RECORD_XLSX,
        "下载记录(带情感)": DOWNLOAD_RECORD_WITH_EMOTION_XLSX,
        "语音情感分析": VOCALTONE_XLSX,
        "语音情感分析(带情感)": VOCALTONE_WITH_EMOTION_XLSX,
        "转录汇总": TRANSCRIPT_SUMMARY_XLSX,
    }


def print_all_paths():
    """打印所有路径配置"""
    print("=" * 70)
    print("路径配置汇总")
    print("=" * 70)
    for name, path in get_all_paths().items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"{exists} {name:20s}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    print_all_paths()

