import json
import os
import time
import argparse
import shortuuid
import concurrent.futures
import tqdm
import tiktoken
from utils import (
    load_questions,
    load_model_answers,
    chat_completion_openai,
    chat_completion_openai_azure,
    chat_completion_anthropic,
    chat_completion_mistral,
    chat_completion_cohere,
    chat_completion_awsbedrock,
    http_completion_gemini,
    get_endpoint,
    make_config,
    reorg_answer_file
)

def reorg_guidance_file(guidance_file):
    """Sort by question id and de-duplication"""
    guidances = {}
    with open(guidance_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            guidances[qid] = l

    qids = sorted(list(guidances.keys()))
    with open(guidance_file, "w") as fout:
        for qid in qids:
            fout.write(guidances[qid])

def generate_guidance(question, ideal_answer, guidance_model, max_tokens, temperature, api_dict, endpoint_info, configs, output_file):
    api_type = endpoint_info["api_type"]
    conv = []

    # Add system prompt from the config file
    if "system_prompt" in configs:
        conv.append({"role": "system", "content": configs["system_prompt"]})
    else:
        conv.append({"role": "system", "content": "You are a helpful assistant."})

    # Prepare the user message using the prompt template from configs
    prompt_args = {
        "question": question['turns'][0]['content'],
        "ideal_answer": ideal_answer['choices'][0]['turns'][0]['content']
    }

    for template in configs["prompt_template"]:
        user_prompt = template.format(**prompt_args)
        conv.append({"role": "user", "content": user_prompt})

    # Select the appropriate chat completion function based on API type
    if api_type == "anthropic":
        output = chat_completion_anthropic(
            model=endpoint_info["model_name"],
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif api_type == "mistral":
        output = chat_completion_mistral(
            model=endpoint_info["model_name"],
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif api_type == "gemini":
        output = http_completion_gemini(
            model=endpoint_info["model_name"],
            message=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif api_type == "azure":
        output = chat_completion_openai_azure(
            model=endpoint_info["model_name"],
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict
        )
    elif api_type == "cohere":
        output = chat_completion_cohere(
            model=endpoint_info["model_name"],
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif api_type == "aws":
        output = chat_completion_awsbedrock(
            model=endpoint_info["model_name"],
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
            api_info=endpoint_info
        )
    else:
        output = chat_completion_openai(
            model=endpoint_info["model_name"],
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict
        )

    # Token counting
    tokenizer = tiktoken.encoding_for_model(guidance_model)
    token_len = len(tokenizer.encode(output))

    # Generate unique IDs and prepare the guidance entry
    guidance_entry = {
        "guidance_id": shortuuid.uuid(),
        "question_id": question["question_id"],
        "ideal_answer_model": configs["ideal_model_id"],  # The model used to generate the ideal answer
        "guidance_model": guidance_model,  # The model used to generate the guidance
        "guidance": output,
        "token_len": token_len,
        "tstamp": time.time(),
    }

    # Write the guidance entry directly to the file
    with open(output_file, "a") as fout:
        fout.write(json.dumps(guidance_entry) + "\n")

def estimate_costs(questions, settings, model_name, avg_output_tokens=550, max_output_tokens=800):
    question_array = [question["turns"][0]["content"] for question in questions]
    tokenizer = tiktoken.encoding_for_model(model_name)
    tokens = [tokenizer.encode(prompt) for prompt in question_array]
    num_input_tokens = sum([len(token) for token in tokens])
    num_questions = len(tokens)

    # gpt-4o rates
    input_multiply = 0.005 / 1000
    output_multiply = 0.015 / 1000

    # Cost estimation
    input_cost = num_input_tokens * input_multiply
    avg_output_cost = num_questions * avg_output_tokens * output_multiply
    max_output_cost = num_questions * max_output_tokens * output_multiply

    print("="*25 + "  Expected Costs (based on GPT-4o)  " + "="*25 + "\n")
    print(f"Expected Input Tokens: \n {num_input_tokens} Tokens in a total of {num_questions} questions\n")
    print(f"Expected Output Tokens: \n {num_questions * avg_output_tokens} Tokens in a total of {num_questions} questions\n")
    print(f"Max Output Tokens: \n {num_questions * max_output_tokens} Tokens in a total of {num_questions} questions\n\n")
    print("-"*25 + "  Resulting in Costs:   " + "-"*25 + "\n")
    print(f"Expected Costs: \n {(input_cost + avg_output_cost):.2f} USD\n")
    print(f"Max. Expected Costs: \n {(input_cost + max_output_cost):.2f} USD\n")

def main(config_file, endpoint_file):
    # Load configurations
    config = make_config(config_file)
    print(config)
    
    # Extract settings from the configuration
    benchmark_name = config["benchmark_name"]
    guidance_model = config["guidance_model"]
    max_tokens = config["max_tokens"]
    temperature = config.get("temperature", 0.0)

    # Paths based on the benchmark name
    base_dir = os.path.join("data", benchmark_name)
    question_file = os.path.join(base_dir, "question.jsonl")
    ideal_answer_dir = os.path.join(base_dir, "model_answer")
    output_file = os.path.join(base_dir, "guidance.jsonl")

    # Load questions and ideal answers
    questions = load_questions(question_file)
    ideal_answers = load_model_answers(ideal_answer_dir)

    # Load existing guidances
    existing_guidances = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as fin:
            for line in fin:
                guidance_data = json.loads(line)
                existing_guidances[guidance_data["question_id"]] = guidance_data

    endpoint_info = make_config(endpoint_file)
    api_dict = get_endpoint(endpoint_info[guidance_model]["endpoints"])

    # Estimate costs
    estimate_costs(questions, config, guidance_model)

    if input("Press Enter to confirm and start generating guidance, or Ctrl+C to cancel...") != "":
        return

    # Determine parallelism
    parallel = endpoint_info[guidance_model].get("parallel", 1)

    # Generate guidance and save to file
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        count = 0
        for question in questions:
            question_id = question["question_id"]

            # Skip if guidance already exists
            if question_id in existing_guidances:
                count += 1
                continue

            ideal_answer = ideal_answers.get(config["ideal_model_id"], {}).get(question_id)

            if not ideal_answer:
                print(f"Ideal answer not found for question_id: {question_id}")
                continue

            future = executor.submit(
                generate_guidance,
                question=question,
                ideal_answer=ideal_answer,
                guidance_model=guidance_model,
                max_tokens=max_tokens,
                temperature=temperature,
                api_dict=api_dict,
                endpoint_info=endpoint_info[guidance_model],
                configs=config,
                output_file=output_file
            )
            futures.append(future)

        if count > 0:
            print(f"{count} number of existing guidances")

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

    # Reorganize the guidance file to ensure it's clean and sorted
    reorg_guidance_file(output_file)

    print("Guidance generation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="config/gen_guidance_config.yaml", help="Path to the configuration file.")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml", help="Path to the endpoint configuration file.")
    
    args = parser.parse_args()

    main(config_file=args.config_file, endpoint_file=args.endpoint_file)