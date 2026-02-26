"""
FinVoc2Vec_all audio - 批量处理所有音频文件（每次10个）
"""
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import torch
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel

# 导入路径配置
from config_paths import MODEL_DIR, AUDIO_DIR, SRT_DIR, PROJECT_ROOT

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_srt_time(time_str: str) -> float:
    """解析SRT时间戳为秒数"""
    time_part, ms_part = time_str.replace(',', '.').split('.')
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    return h * 3600 + m * 60 + s + ms / 1000.0


def parse_srt_file(srt_path: str):
    """解析SRT文件"""
    segments = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n\d+\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        seq_num, start_time, end_time, text = match
        start_sec = parse_srt_time(start_time)
        end_sec = parse_srt_time(end_time)
        text = text.strip()
        if text:
            segments.append((start_sec, end_sec, text))
    
    return segments


def parse_filename(filename: str):
    """解析文件名"""
    parts = filename.replace('.mp3', '').replace('.srt', '').split('_')
    if len(parts) >= 5:
        stock_code = parts[1]
        video_num = parts[-1].replace('video', '')
        return stock_code, video_num
    return "未知", "0"


def load_audio_file(audio_path: str, target_sr: int = 16000):
    """加载音频文件"""
    # 使用 librosa 加载音频（自动转为单声道和目标采样率）
    audio_array, sampling_rate = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio_array, sampling_rate


def extract_audio_segment(audio: np.ndarray, sr: int, start_time: float, end_time: float):
    """提取音频片段"""
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    return audio[start_sample:end_sample]


def process_audio_with_srt(audio_path: str, srt_path: str, model, feature_extractor):
    """处理单个音频文件"""
    target_sr = feature_extractor.sampling_rate
    
    # 加载音频
    audio, sr = load_audio_file(audio_path, target_sr)
    
    # 解析SRT
    segments = parse_srt_file(srt_path)
    
    # 统计
    counts = {'positive': 0, 'neutral': 0, 'negative': 0, 'total_sentences': len(segments)}
    
    # 逐句处理
    for start_time, end_time, text in segments:
        audio_segment = extract_audio_segment(audio, sr, start_time, end_time)
        
        if len(audio_segment) < sr * 0.3:
            continue
        
        inputs = feature_extractor(
            audio_segment,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs['input_values'].to(device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
            logits = outputs['logits'].cpu()
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
        
        label_to_id = model.config.label2id
        max_prob_idx = np.argmax(probs)
        
        if max_prob_idx == label_to_id['positive']:
            counts['positive'] += 1
        elif max_prob_idx == label_to_id['neutral']:
            counts['neutral'] += 1
        elif max_prob_idx == label_to_id['negative']:
            counts['negative'] += 1
    
    return counts


def main():
    print("=" * 80)
    print("FinVoc2Vec - 批量处理所有音频文件（每批10个）")
    print("=" * 80)
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n[1/5] 加载模型...")
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(device)
    model.eval()
    print("✓ 模型加载完成")
    
    # 加载特征提取器
    print("\n[2/5] 加载特征提取器...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
    print("✓ 特征提取器加载完成")
    
    # 获取所有音频文件
    print(f"\n[3/5] 扫描音频文件...")
    audio_files = sorted([
        f for f in os.listdir(AUDIO_DIR)
        if f.endswith('.mp3')
    ])
    
    print(f"✓ 找到 {len(audio_files)} 个音频文件")
    
    # 匹配SRT文件
    print(f"\n[4/5] 匹配字幕文件...")
    matched_pairs = []
    missing_srt = []
    
    for audio_filename in audio_files:
        srt_filename = audio_filename.replace('.mp3', '.srt')
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        srt_path = os.path.join(SRT_DIR, srt_filename)
        
        if os.path.exists(srt_path):
            matched_pairs.append((audio_path, srt_path))
        else:
            missing_srt.append(audio_filename)
    
    print(f"✓ 成功匹配 {len(matched_pairs)} 对音频-字幕文件")
    if missing_srt:
        print(f"⚠ 警告: {len(missing_srt)} 个音频文件缺少对应的SRT字幕文件")
    
    # 批量处理文件（每批10个）
    print(f"\n[5/5] 批量处理音频文件（每批10个）...")
    output_path = os.path.join(PROJECT_ROOT, "FinVoc2Vec_情感分析结果.xlsx")
    results = []
    
    batch_size = 10
    total_batches = (len(matched_pairs) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(matched_pairs))
        batch_pairs = matched_pairs[start_idx:end_idx]
        
        print(f"\n处理批次 {batch_idx + 1}/{total_batches} (文件 {start_idx + 1}-{end_idx}/{len(matched_pairs)})")
        
        for audio_path, srt_path in tqdm(batch_pairs, desc=f"批次 {batch_idx + 1}"):
            filename = os.path.basename(audio_path)
            stock_code, video_num = parse_filename(filename)
            
            try:
                counts = process_audio_with_srt(audio_path, srt_path, model, feature_extractor)
                
                results.append({
                    '股票代码': stock_code,
                    '视频序号': video_num,
                    '积极句子数': counts['positive'],
                    '中立句子数': counts['neutral'],
                    '消极句子数': counts['negative'],
                    '总句子数': counts['total_sentences'],
                    '文件名': filename
                })
                
            except Exception as e:
                print(f"\n  处理失败: {filename} - {e}")
                results.append({
                    '股票代码': stock_code,
                    '视频序号': video_num,
                    '积极句子数': 0,
                    '中立句子数': 0,
                    '消极句子数': 0,
                    '总句子数': 0,
                    '文件名': filename
                })
        
        # 每批处理完后保存结果
        df = pd.DataFrame(results)
        df.to_excel(output_path, index=False)
        print(f"✓ 批次 {batch_idx + 1} 完成，结果已保存")
    
    # 最终统计
    print("\n" + "=" * 80)
    print("全部处理完成！")
    print("=" * 80)
    print(f"输出文件: {output_path}")
    print(f"处理文件数: {len(results)}")
    print(f"\n统计汇总:")
    print(f"  总积极句子数: {df['积极句子数'].sum()}")
    print(f"  总中立句子数: {df['中立句子数'].sum()}")
    print(f"  总消极句子数: {df['消极句子数'].sum()}")
    print(f"  总句子数: {df['总句子数'].sum()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

