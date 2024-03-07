import jsonlines
import json
import argparse

ans_prompt = """You are Medbase, equipped with in-depth knowledge in medicine. Your task is to directly answer the user's <question> in English. In formulating your response, you must thoughtfully reference the <reference text>, ensuring that your reply does not disclose your reliance on <reference text>. Aim to provide a comprehensive and informative response, incorporating relevant insights from <reference text> to best assist the user. Please be cautious to avoid including any content that might raise ethical concerns.

<question>: {question}

<reference text>: {reference}

<reply>: """


def generate_query(data):
    chatgpt_query = ans_prompt.format_map(
        {"question": data["model_answer"], "reference": data["reference"]}
    )
    return chatgpt_query


def Prepare_data(args):
    data = []
    # Read the uploaded JSONl file
    with jsonlines.open(args.input_path, "r") as reader:
        data = list(reader)

    print(f"len:{len(data)}")
    # Convert as required
    jsonl_data = []

    for id, item in enumerate(data):
        jsonl_data.append(
            {
                "id": id,
                "query": generate_query(item),
                "model_answer": "",
                "model_question": item["model_answer"],
                "reference": item["reference"],
            }
        )

    # Save the converted data as a JSONL file
    with open(args.output_path, "w", encoding="utf-8") as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Prepare finished, output to '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for OpenAIGPT generation"
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output JSONL file."
    )
    args = parser.parse_args()
    Prepare_data(args)
