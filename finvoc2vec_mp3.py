import os
import argparse
import numpy as np
import torch
import subprocess
import wave
import tempfile
import shutil
import pandas as pd

from transformers import Wav2Vec2FeatureExtractor
from safetensors.torch import load_file as load_safetensors

# 注意：
# 1. 请将定义 FinVoc2VecConfig 的文件命名为 finvoc2vec_config.py
# 2. 请将定义 FinVoc2Vec 模型类的文件命名为 finvoc2vec_model.py
# 3. 并与本脚本放在同一目录（即 roadshow_downloads）下，或调整下面的导入路径为你实际的包结构
from finvoc2vec_config import FinVoc2VecConfig
from finvoc2vec_model import FinVoc2Vec


def load_audio(path: str, target_sr: int = 16000):
    """
    使用 ffmpeg 将 mp3 转为 wav（单声道 + target_sr），再用 wave+numpy 读成 1D float32 数组。
    依赖：系统已安装 ffmpeg，并可在命令行直接调用 `ffmpeg`。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "tmp.wav")

        cmd = [
            "ffmpeg",
            "-y",              # 覆盖输出
            "-i", path,        # 输入 mp3
            "-ac", "1",        # 单声道
            "-ar", str(target_sr),  # 采样率
            "-f", "wav",
            wav_path,
        ]

        # 调用 ffmpeg，静默运行；如需调试可去掉 stdout/stderr 重定向
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 读取中间 wav
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        # 只支持单声道
        if n_channels != 1:
            raise ValueError(f"期望单声道，实际通道数: {n_channels}")

        # 根据采样位宽解析
        if sampwidth == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sampwidth == 4:
            dtype = np.int32
            max_val = 2147483648.0
        else:
            raise ValueError(f"暂不支持的采样位宽: {sampwidth} 字节（只支持 16/32 bit PCM）")

        audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32) / max_val

        # framerate 理论上等于 target_sr，这里返回 target_sr 保持一致
        return audio, target_sr


def update_excel_with_embedding(
    npy_path: str,
    original_xlsx: str,
    output_xlsx: str | None = None,
    key_col: str = "文件名",
):
    """
    将单个 npy 向量写入下载记录的副本 xlsx 中：新建/更新 negetive/neutral/positive 三列。
    - original_xlsx: 原始下载记录文件路径（不会被修改，只作为模板复制一次）
    - output_xlsx: 结果汇总文件路径；若为 None，则在 original_xlsx 旁边生成 *_带情感结果.xlsx
    - key_col: 用于匹配行的列名，默认 '文件名'，会用 npy 文件名（不含后缀）去匹配
    """
    if output_xlsx is None:
        base, ext = os.path.splitext(original_xlsx)
        output_xlsx = base + "_带情感结果" + ext

    if not os.path.exists(output_xlsx):
        if not os.path.exists(original_xlsx):
            raise FileNotFoundError(f"找不到原始下载记录文件: {original_xlsx}")
        shutil.copyfile(original_xlsx, output_xlsx)

    df = pd.read_excel(output_xlsx)

    # 确保三列存在
    for col in ["negetive", "neutral", "positive"]:
        if col not in df.columns:
            df[col] = None

    emb = np.load(npy_path)
    if emb.shape[0] != 3:
        raise ValueError(f"期望 3 维向量，实际形状为 {emb.shape}，请检查模型输出。")

    neg, neu, pos = map(float, emb.tolist())

    if key_col not in df.columns:
        raise KeyError(f"在 {output_xlsx} 中找不到列 '{key_col}'，请检查列名是否写对。")

    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    mask = df[key_col].astype(str) == str(base_name)

    if not mask.any():
        # 表里没有这行，就追加一行，至少不会丢数据
        new_row = {key_col: base_name, "negetive": neg, "neutral": neu, "positive": pos}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"在表中未找到 {key_col}=='{base_name}' 的行，已在末尾追加一行。")
    else:
        df.loc[mask, "negetive"] = neg
        df.loc[mask, "neutral"] = neu
        df.loc[mask, "positive"] = pos
        print(f"已更新 {mask.sum()} 行: {key_col}=='{base_name}'")

    df.to_excel(output_xlsx, index=False)
    print(f"已保存情感结果到: {output_xlsx}")


def load_model_and_feature_extractor(model_dir: str, device: str = "cpu"):
    """
    从 save_pretrained 的目录加载：
    - FinVoc2VecConfig
    - FinVoc2Vec 模型
    - Wav2Vec2 特征抽取器

    说明：
    这里不再使用 FinVoc2Vec.from_pretrained，而是手动加载 state_dict，
    避免新版本 transformers 在 _finalize_model_loading 中访问 all_tied_weights_keys 报错。
    """
    config = FinVoc2VecConfig.from_pretrained(model_dir)

    # 手动构建模型并加载权重，绕开 from_pretrained 的复杂逻辑
    model = FinVoc2Vec(config)

    # 优先使用 safetensors 格式的权重文件（你当前权重为 model.safetensors）
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.isfile(safetensors_path):
        state_dict = load_safetensors(safetensors_path, device=device)
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location=device)
    else:
        raise FileNotFoundError(
            f"在模型目录中未找到权重文件: {safetensors_path} 或 {bin_path}"
        )
    # 如果 state_dict 是带有其它 key 的字典（如 {"model": ..., "state_dict": ...}），
    # 可以在这里根据实际情况调整：
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    return model, feature_extractor


def extract_finvoc2vec(
    model: FinVoc2Vec,
    feature_extractor: Wav2Vec2FeatureExtractor,
    audio: np.ndarray,
    sr: int,
    device: str = "cpu",
    use_logits: bool = True,
):
    """
    调用 FinVoc2Vec 得到向量：
    - 默认返回 logits（维度 = num_labels）
    - 如需用 pooled hidden_states，可在这里改写
    """
    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    if use_logits:
        emb = outputs["logits"].squeeze(0).cpu().numpy()
    else:
        # 示例：如想用 pooled hidden_states，可参考：
        # last_hidden = outputs["hidden_states"][-1]  # [B, T, H]
        # pooled = model.merged_strategy(last_hidden, mode=model.pooling_mode)  # [B, H]
        # emb = pooled.squeeze(0).cpu().numpy()
        emb = outputs["logits"].squeeze(0).cpu().numpy()

    return emb


def process_single_file(
    mp3_path: str,
    out_dir: str,
    model: FinVoc2Vec,
    feature_extractor: Wav2Vec2FeatureExtractor,
    sample_rate: int,
    device: str,
    use_logits: bool = True,
    original_xlsx: str | None = None,
    output_xlsx: str | None = None,
    xlsx_key_col: str = "文件名",
):
    os.makedirs(out_dir, exist_ok=True)

    audio, sr = load_audio(mp3_path, target_sr=sample_rate)
    emb = extract_finvoc2vec(
        model=model,
        feature_extractor=feature_extractor,
        audio=audio,
        sr=sr,
        device=device,
        use_logits=use_logits,
    )

    base = os.path.splitext(os.path.basename(mp3_path))[0]
    out_path = os.path.join(out_dir, base + ".npy")
    np.save(out_path, emb)
    print(f"已处理: {mp3_path} -> {out_path}")

    # 如果提供了下载记录 xlsx，则顺便写入情感结果
    if original_xlsx is not None:
        try:
            update_excel_with_embedding(
                npy_path=out_path,
                original_xlsx=original_xlsx,
                output_xlsx=output_xlsx,
                key_col=xlsx_key_col,
            )
        except Exception as e:
            print(f"写入 Excel 时出错: {e}")


def process_path(
    input_path: str,
    out_dir: str,
    model: FinVoc2Vec,
    feature_extractor: Wav2Vec2FeatureExtractor,
    sample_rate: int,
    device: str,
    use_logits: bool = True,
    original_xlsx: str | None = None,
    output_xlsx: str | None = None,
    xlsx_key_col: str = "文件名",
):
    if os.path.isfile(input_path):
        process_single_file(
            mp3_path=input_path,
            out_dir=out_dir,
            model=model,
            feature_extractor=feature_extractor,
            sample_rate=sample_rate,
            device=device,
            use_logits=use_logits,
            original_xlsx=original_xlsx,
            output_xlsx=output_xlsx,
            xlsx_key_col=xlsx_key_col,
        )
    else:
        for root, _, files in os.walk(input_path):
            for name in files:
                if name.lower().endswith(".mp3"):
                    mp3_path = os.path.join(root, name)
                    process_single_file(
                        mp3_path=mp3_path,
                        out_dir=out_dir,
                        model=model,
                        feature_extractor=feature_extractor,
                        sample_rate=sample_rate,
                        device=device,
                        use_logits=use_logits,
                        original_xlsx=original_xlsx,
                        output_xlsx=output_xlsx,
                        xlsx_key_col=xlsx_key_col,
                    )


def main():
    parser = argparse.ArgumentParser(
        description="对 mp3 文件进行 FinVoc2Vec 处理"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="save_pretrained 保存的 FinVoc2Vec 模型目录，例如 D:\\project\\Friendly-Downloader\\roadshow_downloads\\FinVoc2Vec_code",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入：单个 mp3 文件路径，或包含 mp3 的目录路径",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="输出向量目录（每个 mp3 对应一个 .npy）",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="目标采样率，需与训练时一致",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="设备：cpu 或 cuda，例如 cuda:0",
    )
    parser.add_argument(
        "--use_logits",
        action="store_true",
        help="使用 logits 作为向量（默认建议打开）",
    )
    # Excel 相关参数：在下载记录的副本中写入情感结果
    parser.add_argument(
        "--xlsx",
        default=r"D:\project\Friendly-Downloader\output_Vocaltone.xlsx",
        help="原始下载记录 xlsx 路径（不会被修改，会在其基础上生成带情感结果的副本）",
    )
    parser.add_argument(
        "--xlsx_output",
        default=None,
        help="输出汇总 xlsx 路径；默认在原始文件旁生成 *_带情感结果.xlsx",
    )
    parser.add_argument(
        "--xlsx_key_col",
        default="文件名",
        help="用于匹配行的列名（例如 '文件名'），将用 npy 文件名（不含扩展名）去匹配这一列",
    )

    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换为 CPU")
        device = "cpu"

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"模型目录不存在: {args.model_dir}")

    model, feature_extractor = load_model_and_feature_extractor(args.model_dir, device=device)

    process_path(
        input_path=args.input,
        out_dir=args.out_dir,
        model=model,
        feature_extractor=feature_extractor,
        sample_rate=args.sample_rate,
        device=device,
        use_logits=args.use_logits,
        original_xlsx=args.xlsx,
        output_xlsx=args.xlsx_output,
        xlsx_key_col=args.xlsx_key_col,
    )


if __name__ == "__main__":
    main()


