import json
import yaml
import argparse
import os
import re
import concurrent.futures
import tiktoken

from tqdm import tqdm

from utils import (
    prepare_batch_data,
    save_batch_result,
    chat_completion_openai,
    chat_completion_openai_azure,
    chat_completion_openai_azure_batched,
    chat_completion_anthropic,
    load_questions,
    load_model_answers,
    get_endpoint,
    make_config,
    OPENAI_MODEL_LIST,
)


def get_score(judgment, pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False


# get answer from model
def get_answer(model, conv, temperature, max_tokens, endpoint_dict=None):
    api_dict = get_endpoint(endpoint_dict["endpoints"])

    if endpoint_dict["api_type"] == "anthropic":
        output = chat_completion_anthropic(model, conv, temperature, max_tokens)
    elif endpoint_dict["api_type"] == "azure":
        output = chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict)
    else:
        output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict)
    return output



def batch_judgement(models, questions, model_answers, ref_answers, configs, output_files, endpoint_info, args, pattern):
    batch_data = []
    for model in models:
        for question in questions:
            question_id = question["question_id"]
            if question_id not in model_answers[model]:
                print(f"Skipping question {question_id} for model {model} as answer is missing.")
                continue

            # Call prepare_batch_data with explicit arguments
            tasks = prepare_batch_data(
                question=question,
                answer=model_answers[model].get(question_id),
                reference=ref_answers,
                baseline_answer=model_answers[configs["baseline_model"]].get(question_id) if configs["baseline"] else None,
                configs=configs,
                endpoint_dict=endpoint_info
            )
            
            if tasks:
                batch_data += tasks

            if args.test_only:
                break

            if not batch_data:
                raise ValueError("No batch data was prepared. Check if all necessary inputs are provided.")

        # Write batch data to a file
        batch_file_name = "azure_batch_input.jsonl"
        with open(batch_file_name, 'w') as f:
            for task in batch_data:
                f.write(json.dumps(task) + '\n')

        #print(endpoint_info["endpoints"][0])
    
        # Call the Azure batch processing function
        batch_results = chat_completion_openai_azure_batched(batch_file_name, endpoint_info["endpoints"][0])

        # Save results
        for result in batch_results:
            save_batch_result(result, output_files[model], questions)


def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    output_file = args["output_file"]
    model = configs["judge_model"]

    num_games = 2 if configs["pairwise"] else 1

    output = {
        "question_id": question["question_id"],
        "model": answer["model_id"],
        "judge": model,
        "games": []
        }

    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
            prompt_args = {}

            for i, turn in enumerate(question["turns"]):
                prompt_args[f"question_{i+1}"] = turn["content"]
            base = 1

            if baseline:
                if game % 2 == 1: # swap position
                    answer, baseline = baseline, answer

                for i, turn in enumerate(baseline["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+1}"] = turn["content"]
                    base += 1
            if answer:
                for i, turn in enumerate(answer["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+base}"] = turn["content"]

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]
            
            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        for _ in range(configs['number_of_judgment_attempts']):
            new_judgment = get_answer(
                endpoint_info["model_name"],
                conv,
                configs["temperature"],
                configs["max_tokens"],
                args["endpoint_dict"],
            )

            judgment += ("\n" + new_judgment)

            score, try_again = get_score(judgment, args["regex_pattern"])

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})

        result = {
            "user_prompt": conv[1]["content"],
            "judgment": judgment,
            "score": score
        }
        output["games"].append(result)

    with open(output_file, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/judge_config.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    parser.add_argument("--test-only", action="store_true", help="Process only one batch")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
          + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}')

    if configs["regex_pattern"]:
        pattern = re.compile(configs["regex_pattern"])

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")
    ref_answer_dir = os.path.join("data", configs["bench_name"], "reference_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)
    
    models = [model for model in configs["model_list"]]
        
    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]
    
    output_files = {}
    output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    endpoint_info = endpoint_list[configs["judge_model"]]

    print("Estimating Costs...")

    num_questions = len(questions)

    if configs["judge_model"] in OPENAI_MODEL_LIST:
        tokenizer = tiktoken.encoding_for_model(configs["judge_model"])
    else: 
        tokenizer = tiktoken.encoding_for_model("gpt-4")

    num_games = 2 if configs["pairwise"] else 1
    
    system_prompt = configs["system_prompt"]
    num_system_prompt_tokens = len(tokenizer.encode(system_prompt)*num_questions)

    question_array = [question["turns"][0]["content"] for question in questions]
    tokens = [tokenizer.encode(prompt) for prompt in question_array]
    num_question_tokens = sum([len(token) for token in tokens])

    # Based on the Number of Tokens for 1 Model Judgement with baseline (gpt-4-0613)
    num_judge_tokens = 320000 if configs["baseline"] else 160000

    num_answer_tokens = []
    question_ids = [question["question_id"] for question in questions]

    if configs["baseline"]:
        baseline_answers = [model_answers[configs["baseline_model"]][question_id] for question_id in question_ids]
        baseline_answers = [answer["choices"][0]["turns"][0]["content"] for answer in baseline_answers]
        baseline_tokens = [tokenizer.encode(answer) for answer in baseline_answers]
        num_baseline_tokens = sum([len(token) for token in baseline_tokens])

    for model in models:
        answers = [model_answers[model][question_id] for question_id in question_ids]
        answers = [answer["choices"][0]["turns"][0]["content"] for answer in answers]
        answer_tokens = [tokenizer.encode(answer) for answer in answers]
        if configs["baseline"]:
            num_answer_tokens.append(sum([len(token) for token in answer_tokens]) + num_baseline_tokens)
        else:
            num_answer_tokens.append(sum([len(token) for token in answer_tokens]))
    
    total_number_input_tokens = (num_question_tokens + num_system_prompt_tokens + sum(num_answer_tokens)) * num_games
    total_number_output_tokens = num_judge_tokens * num_games

    if args.test_only:
        batch_share = endpoint_info["parallel"] / num_questions
    else: 
        batch_share = 1

    total_number_input_tokens = total_number_input_tokens * batch_share
    total_number_output_tokens = total_number_output_tokens * batch_share


    if endpoint_info["api_type"] == "azure_batched":
        # bacth rates 
        input_muliply = (0.005 / 1000) * 0.5
        output_muliply = (0.015 / 1000) * 0.5

    else:
        # gpt-4o rates
        input_muliply = 0.005 / 1000
        output_muliply = 0.015 / 1000

    judge_input_cost = total_number_input_tokens * input_muliply
    judge_output_cost = total_number_output_tokens * output_muliply

    print("="*25 + "  Expected Costs (based on GPT-4o)  " + "="*25 + "\n")
    print(f"Expected Input Tokens: \n {total_number_input_tokens} Tokens in a total of {int(num_questions * num_games * batch_share)} games\n")
    print(f"Expected Output Tokens: \n {total_number_output_tokens} Tokens in a total of {int(num_questions * num_games * batch_share)} games\n")
    print("-"*25 + "  Resulting in Costs:   " + "-"*25 + "\n")
    print(f"Costs for Input Tokens: \n {judge_input_cost:.2f} USD   --   Costs for Output Tokens {judge_output_cost:.2f} USD\n")
    print(f"Expected total Costs: \n {(judge_input_cost + judge_output_cost):.2f} USD\n")

    input("Press Enter to confirm...")
    print("Starting to generate judgement...")

    if endpoint_info["api_type"] == "azure_batched":
        batch_judgement(models, questions, model_answers, ref_answers, configs, output_files, endpoint_info, args, pattern)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_info["parallel"]) as executor:
            futures = []
            for model in models:
                count = 0
                for question in questions:
                    question_id = question["question_id"]

                    kwargs = {}
                    kwargs["question"] = question
                    if model in model_answers and not question_id in model_answers[model]:
                        print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                        continue

                    if model in existing_judgments and question_id in existing_judgments[model]:
                        count += 1
                        continue

                    kwargs["answer"] = model_answers[model][question_id]
                    if ref_answers:
                        kwargs["reference"] = [ref_answer[question_id] for ref_answer in ref_answers]
                        assert len(kwargs["reference"]) == len(configs["ref_model"])
                    else:
                        kwargs["reference"] = None
                    if configs["baseline"]:
                        kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][question_id]
                    else:
                        kwargs["baseline_answer"] = None
                    kwargs["configs"] = configs
                    kwargs["endpoint_dict"] = endpoint_info
                    kwargs["output_file"] = output_files[model]
                    kwargs["regex_pattern"] = pattern
                    future = executor.submit(judgment, **kwargs)
                    futures.append(future)

                if count > 0:
                    print(f"{count} number of existing judgments")

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()