import os
import glob
import subprocess


def main():
    # 项目根目录
    project_dir = r"D:\project\Friendly-Downloader"
    model_dir = os.path.join(project_dir, "finvoc2vec_code")
    audio_dir = os.path.join(project_dir, "roadshow_downloads", "audio")
    out_dir = project_dir  # 输出到项目根目录

    # 找出 audio 目录下按文件名排序的所有mp3
    mp3_files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))

    if not mp3_files:
        print(f"在目录中没有找到 mp3 文件: {audio_dir}")
        return

    print("将要处理的 5 个文件（不足 5 个则全部处理）：")
    for f in mp3_files:
        print(" -", f)

    for mp3_path in mp3_files:
        cmd = [
            "python",
            os.path.join(model_dir, "finvoc2vec_mp3.py"),
            "--model_dir",
            model_dir,
            "--input",
            mp3_path,
            "--out_dir",
            out_dir,
            "--sample_rate",
            "16000",
            "--device",
            "cpu",
            "--use_logits",
        ]
        print("\n运行命令：", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


