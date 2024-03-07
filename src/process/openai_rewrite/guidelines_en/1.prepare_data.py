import jsonlines
import json
import argparse

query_prompt = """Please create a <question> that closely aligns with the provided <text>. Ensure that the <question> is formulated in English and does not explicitly reference the text. You may incorporate specific scenarios or contexts in the <question>, allowing the <text> to serve as a comprehensive and precise answer.

<text>: {text}

<question>: """


def generate_query(data):
    chatgpt_query = query_prompt.format_map({"text": data[0]})
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
                "reference": item[0],
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
