import jsonlines
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import argparse
from OpenAIGPT import OpenAIGPT


def OpenAIGPT_datagen(args):
    igpt = OpenAIGPT(model_name=args.model_name, keys_path=args.keys_path)

    def process_item(item):
        content = igpt(item["query"])
        item["model_answer"] = content
        return item

    output_path = args.output_path
    input_path = args.input_path

    # Collect the IDs of processed items in the output file
    processed_ids = set()
    if os.path.exists(output_path):
        with jsonlines.open(output_path, "r") as f:
            for item in f:
                processed_ids.add(item.get("id", None))

    # Collect unprocessed items
    items_to_process = []

    with jsonlines.open(input_path, "r") as reader:
        for item in reader:
            item_id = item.get("id", None)
            if item_id is not None and item_id in processed_ids:
                continue
            items_to_process.append(item)

    # Multi-threaded parallel processing
    with jsonlines.open(
        output_path, "a" if os.path.exists(output_path) else "w"
    ) as writer:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_item, item): item for item in items_to_process
            }

            # Use tqdm to display progress
            for future in tqdm(
                futures, total=len(items_to_process), desc="Processing items"
            ):
                item = future.result()
                writer.write(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files concurrently.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        help="Name of the OpenAIGPT model to use.",
    )
    parser.add_argument(
        "--keys_path",
        type=str,
        required=True,
        help="API key for the OpenAIGPT service.",
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of workers for concurrent processing.",
    )

    args = parser.parse_args()
    OpenAIGPT_datagen(args)
