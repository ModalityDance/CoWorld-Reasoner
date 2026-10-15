"""Download the development data for evaluation and co-evolution."""

import json
from pathlib import Path

from datasets import Image, List, load_dataset


DATASET_ID = "ModalityDance/CoWorld-Reasoner-Dev"
TASK_FIELDS = (
    "sample_id", "benchmark", "task_type", "answer_type",
    "question", "options", "image_paths",
)
ANSWER_FIELDS = (
    "sample_id", "benchmark", "answer_type", "question", "answer", "score_spec",
)


def save_data(dataset, root: Path) -> None:
    data_root = root / "data" / "dev120"
    answer_file = root / "scorer_bundle" / "data" / "dev120_evaluator_only.jsonl"
    dataset = dataset.cast_column("images", List(Image(decode=False)))
    tasks, answers = [], []
    for example in dataset:
        paths, images = example["image_paths"], example["images"]
        if len(paths) != len(images):
            raise ValueError(f"Image count mismatch: {example['sample_id']}")
        for relative, image in zip(paths, images):
            destination = (data_root / relative).resolve()
            if not destination.is_relative_to(data_root.resolve()):
                raise ValueError(f"Invalid image path: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(image["bytes"])
        tasks.append({key: example[key] for key in TASK_FIELDS})
        answer = {key: example[key] for key in ANSWER_FIELDS}
        if example.get("official_problem") is not None:
            answer["official_problem"] = example["official_problem"]
        answers.append(answer)
    data_root.mkdir(parents=True, exist_ok=True)
    answer_file.parent.mkdir(parents=True, exist_ok=True)
    files = ((data_root / "public.jsonl", tasks), (answer_file, answers))
    for path, rows in files:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(tasks)} development examples to {data_root}")


if __name__ == "__main__":
    save_data(load_dataset(DATASET_ID, split="train"), Path(__file__).resolve().parent)
