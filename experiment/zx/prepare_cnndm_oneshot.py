from pathlib import Path
from datasets import load_dataset

OUTPUT = Path(r"C:\Users\zhuxu\Desktop\cnn_dm_oneshot_prompts")
OUTPUT.mkdir(parents=True, exist_ok=True)

# 新版 datasets 使用Parquet仓库，避免旧dataset script报错
repo = "abisee/cnn_dailymail"

train = load_dataset(repo, "3.0.0", split="train")
test = load_dataset(repo, "3.0.0", split="test")

# 与LayerSkip官方data.py一致
shot = train.shuffle(seed=42).select(range(1))[0]

shot_prompt = (
    "Article: "
    + shot["article"]
    + "\nSummary: "
    + shot["highlights"].replace("\n", "")
    + "\n\n"
)

for i, example in enumerate(test.select(range(10))):
    prompt = (
        shot_prompt
        + "Article: "
        + example["article"]
        + "\nSummary:"
    )

    path = OUTPUT / f"cnn_dm_oneshot_{i:02d}.txt"
    path.write_text(prompt, encoding="utf-8")

print(f"Generated 10 official-format one-shot prompts in {OUTPUT}")