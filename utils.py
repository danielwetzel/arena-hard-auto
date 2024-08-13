import os
import json
import time
import yaml
import random
import requests

from typing import Optional
from glob import glob

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4o",
)


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}

def load_guidance(guidance_file: str):
    """Load guidance from a file."""
    guidance = {}
    with open(guidance_file, "r") as fin:
        for line in fin:
            if line:
                line = json.loads(line)
                guidance[line["question_id"]] = line
    return guidance

def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers



def prepare_batch_data(question, answer, reference, baseline_answer, configs, endpoint_dict):
    batch_data = []

    conv = [{"role": "system", "content": configs["system_prompt"]}]

    for template in configs["prompt_template"]:
        prompt_args = {}

        for i, turn in enumerate(question["turns"]):
            prompt_args[f"question_{i+1}"] = turn["content"]
        base = 1

        if baseline_answer and "choices" in baseline_answer and baseline_answer["choices"]:
            for i, turn in enumerate(baseline_answer["choices"][0]["turns"]):
                prompt_args[f"answer_{i+1}"] = turn["content"]
                base += 1

        if answer and "choices" in answer and answer["choices"]:
            for i, turn in enumerate(answer["choices"][0]["turns"]):
                prompt_args[f"answer_{i+base}"] = turn["content"]

        if reference:
            for j, ref_answer in enumerate(reference):
                if "choices" in ref_answer and ref_answer["choices"]:
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]

        user_prompt = template.format(**prompt_args)
        conv.append({"role": "user", "content": user_prompt})

    # Ensure the question has a valid question_id
    if "question_id" not in question or not question["question_id"]:
        raise ValueError(f"Question is missing a valid question_id: {question}")

    batch_task = {
        "custom_id": question["question_id"],  # Use question_id as custom_id
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": endpoint_dict["model_name"],
            "messages": conv
        }
    }

    batch_data.append(batch_task)
    return batch_data



def save_batch_result(result, output_file, questions):
    question_id = result["custom_id"]
    if question_id not in questions:
        raise ValueError(f"Result refers to an unknown question_id: {question_id}")
    
    judgment = result["response"]["body"]["choices"][0]["message"]["content"]

    # Save the result in the correct output file
    output_entry = {
        "question_id": question_id,
        "judgment": judgment,
        "tstamp": time.time()
    }

    with open(output_file, 'a') as f:
        f.write(json.dumps(output_entry) + '\n')



def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
    
    return output


def chat_completion_openai_azure(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint = api_base,
        api_key= api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output



def chat_completion_openai_azure_batched(batch_file_name, api_dict):
    from openai import AzureOpenAI
    import time
    import datetime
    import json

    client = AzureOpenAI(
        azure_endpoint=api_dict["api_base"],
        api_key=api_dict["api_key"],
        api_version=api_dict["api_version"],
    )

    # Upload the batch file
    file_response = client.files.create(
        file=open(batch_file_name, "rb"),
        purpose="batch"
    )

    
    file_id = file_response.id

    # Wait for the file to be processed
    status = "pending"
    while status != "processed":
        time.sleep(15)
        file_response = client.files.retrieve(file_id)
        status = file_response.status
        print(f"{datetime.datetime.now()} File Id: {file_id}, Status: {status}")

    # Submit the batch job
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",
        completion_window="24h",
    )
    batch_id = batch_response.id
    print(f"Batch ID: {batch_id}")

    # Track the batch job progress
    status = "validating"
    while status not in ("completed", "failed", "canceled"):
        time.sleep(60)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

    # Retrieve the output file
    file_response = client.files.content(batch_response.output_file_id)
    raw_responses = file_response.text.strip().split('\n')

    batch_results = []
    for raw_response in raw_responses:
        json_response = json.loads(raw_response)
        batch_results.append(json_response)

    return batch_results


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens):
    api_key = os.environ["GEMINI_API_KEY"]
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{
                    "parts":[
                        {"text": message}
                    ]
                }],
                "safetySettings": safety_settings,
                "generationConfig":{
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output
    


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


def chat_completion_awsbedrock(model, messages, temperature, max_tokens, api_dict=None, api_info=None):
    from transformers import  AutoTokenizer
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config

    if 'api_secret_key' in api_dict:
        aws_key = api_dict["api_key"]
        aws_secret_key = api_dict["api_secret_key"]
    else:
        aws_key = os.environ["AWS_KEY"]
        aws_secret_key = os.environ["AWS_SECRET_KEY"]

    if 'aws_region' in api_dict:
        aws_region = api_dict["aws_region"]
    else:
        aws_region = 'us-west-2'

    # Set the timeout for the Boto Client to 10 minutes - The default timeout raised "Read timeout" Exeptions for longer questions
    boto_config = Config(read_timeout=600)

    # initialize the bedrock runtime client
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', 
                            region_name=aws_region, 
                            aws_access_key_id=aws_key, 
                            aws_secret_access_key=aws_secret_key, 
                            config=boto_config)


    # llama models in AWS Bedrock don't use a json-like chat template
    if 'model_type' in api_info:
        model_type = api_info["model_type"]
    else: 
        model_type = 'llama-3.1'
    
    # Llama Models Require the Prompt in one String
    if 'llama' in model_type:
        if model_type == 'llama-3.1':
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct", padding_side="left")
        elif model_type == 'llama-3':
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", padding_side="left")
        else:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", padding_side="left")

        tokenizer.pad_token = tokenizer.bos_token
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        body = json.dumps({
            "prompt": prompt,
            "max_gen_len":max_tokens,
            "temperature":temperature,
            "top_p":0.8
        })

        try:
            response_json = bedrock_runtime.invoke_model(body=body, 
                                                modelId=model, 
                                                accept="application/json", 
                                                contentType="application/json")

            response = json.loads(response_json['body'].read())
            output = response["generation"]

        except (ClientError, Exception) as e:
            error = f"ERROR: Can't invoke '{model}'. Reason: {e}"
            print(error)
            output = API_ERROR_OUTPUT


    # Claude Models Require the Anthropic Version
    elif 'claude' in model_type:
        if 'anthropic_version' in api_info:
            anthropic_version = api_info["anthropic_version"]
        else:
            anthropic_version = 'bedrock-2023-05-31'

        body = json.dumps({
            "anthropic_version": anthropic_version,
            "max_tokens": max_tokens,
            "temperature":temperature,
            "messages": messages
        })
        try:
            response_json = bedrock_runtime.invoke_model(body=body, 
                                                modelId=model, 
                                                accept="application/json", 
                                                contentType="application/json")

            response = json.loads(response_json['body'].read())
            output = response['choices'][0]['message']['content']

        except (ClientError, Exception) as e:
            error = f"ERROR: Can't invoke '{model}'. Reason: {e}"
            print(error)
            output = API_ERROR_OUTPUT
    
    # For a few of the other models there might be a different format as well. This needs to be checked
    else: 
        body = json.dumps({
            "max_tokens": max_tokens,
            "temperature":temperature,
            "messages": messages
        })
        try:
            response_json = bedrock_runtime.invoke_model(body=body, 
                                                modelId=model, 
                                                accept="application/json", 
                                                contentType="application/json")

            response = json.loads(response_json['body'].read())
            output = response['choices'][0]['message']['content']

        except (ClientError, Exception) as e:
            error = f"ERROR: Can't invoke '{model}'. Reason: {e}"
            print(error)
            output = API_ERROR_OUTPUT


    return output



def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
