from datasets import load_dataset
from pathlib import Path
import json

output_dir = Path(r"C:\Users\zhuxu\Desktop\cnn_dm_prompts")
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "parquet",
    data_files={
        "test": (
            "https://huggingface.co/datasets/abisee/cnn_dailymail/"
            "resolve/main/3.0.0/test-00000-of-00001.parquet"
        )
    },
    split="test",
)

dataset = dataset.shuffle(seed=221117192)

selected = []

for row in dataset:
    article = row["article"].strip()
    word_count = len(article.split())

    # Avoid articles that are extremely short or may exceed the phone context.
    if not 400 <= word_count <= 1400:
        continue

    selected.append(row)

    if len(selected) == 100:
        break

with (output_dir / "manifest.jsonl").open(
    "w", encoding="utf-8"
) as manifest:
    for index, row in enumerate(selected):
        sample_id = f"cnn_dm_{index:02d}"

        prompt = (
            "Summarize the following news article in three concise bullet "
            "points. Only include information supported by the article.\n\n"
            "Article:\n"
            f"{row['article'].strip()}\n\n"
            "Summary:\n"
        )

        (output_dir / f"{sample_id}.txt").write_text(
            prompt,
            encoding="utf-8",
        )

        manifest.write(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "dataset_id": row["id"],
                    "word_count": len(row["article"].split()),
                    "reference": row["highlights"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

print(f"Generated {len(selected)} prompts in {output_dir}")