# Importation rules.
# 1st, import standard library modules.
# 2nd, import 3rd party library modules.
# 3rd, import project-local modules.
# 4th, violate the above order only when violations are unavoidable.

import abc
import argparse
import codecs
import json
import logging
import multiprocessing
import os
import os.path
import random
from functools import partial
from time import sleep
from typing import override

import backoff
import langchain.messages
import langchain_core.utils.uuid as _uuid
import mcma.cmds.agent_v1.serve as _serve
import requests
from openai import APIStatusError, RateLimitError
from tqdm import tqdm

from ugmb_vlyc.utils import *

# Declaration rules.
# 1st, declare local aliases.
# 2nd, declare private module attributes.
# 3rd, declare public module attributes.
# 4th, violate the above order only when violations are unavoidable.

random_seed = 0
random.seed(random_seed)
_HumanMessage = langchain.messages.HumanMessage
_AIMessage = langchain.messages.AIMessage
_ToolMessage = langchain.messages.ToolMessage

SUBJECTS = [
    "abstract_algebra",
    "combinatorics",
    "statistics",
    "geometry",
    "set_theory_and_logic",
    "probability",
    "arithmetic",
    "trigonometry",
    "complex_analysis",
    "differential_equations",
    "calculus_single_variable",
    "linear_algebra",
    "calculus_multivariable",
    "algebra",
    "number_theory",
    "financial_mathematics"
]
VERSION_COUNT = 3


class PredictorBase:
    @abc.abstractmethod
    def generate(self, prompts):
        pass

    def perform_inference(self, batch):
        # 20240424 cot-sc is disabled please use tag v1.0 for cot-sc evaluation
        return self.post_process(self.generate(batch))

    def post_process(self, completions):
        return completions

    @abc.abstractmethod
    @override
    def __str__(self):
        return str(super())


@backoff.on_exception(backoff.constant, RateLimitError, interval=60, logger=logging.getLogger())  # 超过qps暂停60秒
def get_gpt_response(prompt, model='gpt-4-0613', max_tokens=2048, stop_tokens=None, temperature=0.0):
    if stop_tokens is None:
        stop_tokens = []
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "stop": stop_tokens,
            "temperature": temperature
        }
        if 'o1' in model:

            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": max_tokens,
                "temperature": temperature
            }
        # result = client.request("POST", "/v1", payload, headers)#.getresponse().read()
        try:
            result = requests.request(method="POST", url=client, headers=headers, json=payload).json()
        except:
            sleep(10.0)
            print("request too fast, sleep 3s.")
            return get_gpt_response(prompt, model)
        try:
            # print(result['choices'][0]['message']['content'])
            return result['choices'][0]['message']['content']
        except:
            return None
    except APIStatusError as e:
        if e.status_code in [450, 451]:  # 保时洁检测
            return "内容违规，未通过保时洁检查。"
        if e.status_code == 400:
            return "输入有误，可能是上下文长度不够"
        if e.status_code == 429:
            raise e
        if e.status_code == 408:
            raise e
        print(e.message)
        return get_gpt_response(prompt, model)


def initialize_client(openai_api_key, openai_base_url):
    global client
    global headers
    # client = http.client.HTTPSConnection(openai_base_url)
    client = openai_base_url
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {openai_api_key}",
    }


class OpenAIPredictor(PredictorBase):
    def __init__(self, model, nproc=16):
        self.gpt_series = model
        self.openai_api_key = os.environ["OPENAI_BASE_URL"]
        self.openai_base_url = os.environ["OPENAI_API_KEY"]
        self.nproc = nproc

    @override
    def generate(self, prompts):
        with multiprocessing.Pool(
            processes=self.nproc,
            initializer=initialize_client, initargs=(
                self.openai_api_key, self.openai_base_url)
        ) as pool:
            completions = list(tqdm(
                pool.imap(
                    partial(
                        get_gpt_response,
                        model=self.gpt_series
                    ),
                    prompts
                ), total=len(prompts), desc="Performing OpenAI predictions"
            ))
        return completions

    @override
    def __str__(self):
        return f"<OpenAIPredictor>(model={self.gpt_series})"


def generate(
        model_path,
        dataset_path,
        output_path,
        version=0,
        prompt="raw",
        nproc=16,
        test_part=None,
        test_count=1,
        random_seed=None,
    ):
    """
    Generate model's response using vLLM. 

    Parameters:
        model_path: Path to your model
        dataset_path: Path to your dataset. We provide some data in eval/input as examples. If you wat to use your own dataset, please convert the dataset to json format with a list of test samples. Each sample contains at least "answer" and "type". 
        prompt: The prompt to evaluate. We provide some prompts in prompt.json and you can add new prompts following our examples.
        output_path: Path to save the results.
        tensor_parallel_size: The number of GPUs you want to use. 
        test_part: The range of data to be processed. [low, high]. If left None, will process all the data. 
        test_count: Number of problems to draw from the dataset.
        random_seed: Pseudo-random number generator seed. None stands for non-determinstic pseudo-random number generator behavior.
    """

    try:
        random_seed = int(random_seed)
    except Exception as _:
        random_seed = None

    random.seed(random_seed)

    # Load test data
    with open(dataset_path, 'r') as file:
        test_dataset_list = json.load(file)

    data_count = len(test_dataset_list)
    data_indexes = list(range(data_count))
    data_count_to_use = int(min(test_count, data_count))
    data_indexes_to_use = random.choices(data_indexes, k=data_count_to_use)

    exists = []

    # deal with version
    all_version_indexes = list(range(VERSION_COUNT))
    version_indexes = []
    if version == -1:
        version_indexes = all_version_indexes
    else:
        version_indexes = [version - 1]

    test_dataset_list_new = []
    for index in version_indexes:
        if not os.path.exists(output_path.split(".js")[0] + f"_v{index + 1}.json"):
            for index_ in data_indexes_to_use:
                item = test_dataset_list[index_]
                item_copy = item.copy()
                item_copy['problem'] = item['problem_v' + str(index + 1)]
                item_copy['answer'] = item['answer_v' + str(index + 1)]
                item_copy['version'] = index + 1
                item_copy['answer_type'] = item['answer_type_v'+str(index + 1)]
                item_copy['options'] = item['options_v'+str(index + 1)]
                test_dataset_list_new.append(item_copy)
            print(f"VERSION: {index + 1} LOADED!")
        else:
            with open(output_path.split(".js")[0] + f"_v{index + 1}.json", "rt", encoding="utf-8") as file:
                exists += json.load(file)
            print(f"VERSION: {index + 1} already exists!")
    for item in test_dataset_list_new:
        for index in all_version_indexes:
            try:
                del item['problem_v'+str(index + 1)]
                del item['answer_v'+str(index + 1)]
                del item['answer_type_v'+str(index + 1)]
                del item['options_v'+str(index + 1)]
            except:
                pass
    print(f"ALL VERSIONS LOADED!")
    test_dataset_list = test_dataset_list_new
    if len(test_dataset_list) == 0:
        print(f"ALL VERSIONS ALREADY EVALUATED, WRITING TO {output_path}!")
        with open(output_path, 'w+t', encoding="utf-8") as file:
            json.dump(exists, file, indent=4)

    # Load prompt
    with open("prompt.json", 'r', encoding="utf-8") as file:
        prompt_dict = json.load(file)[prompt]

    # TODO: deal with inference prompt
    test_dataset_list = make_prompt(prompt_dict, test_dataset_list)

    if test_part is not None:
        test_dataset_list = test_dataset_list[test_part[0]:test_part[1]]
    batch_size = len(test_dataset_list)

    for items in test_dataset_list:
        items["completion"] = ""

    _serve.args_overridden = True
    _serve.args = []
    _serve.main()
    agent = _serve.agent

    for index_global in range(len(test_dataset_list) // batch_size):
        # Batch Construct
        if batch_size * (index_global+2) > len(test_dataset_list):
            curr_data_list = test_dataset_list[batch_size * index_global:]
        else:
            curr_data_list = test_dataset_list[batch_size * index_global:batch_size * (index_global+1)]

        # Inference
        prompts = [item['prompt'] for item in curr_data_list]
        completions = []
        for index_, prompt_ in enumerate(prompts):
            input_messages = _HumanMessage(content=prompt_)
            print(f"Will do agent invocation {index_ + 1}/{len(prompts)}")
            thread_id = _uuid.uuid7()
            config={"configurable": {"thread_id": str(thread_id)}}
            response = agent.invoke(
                {"messages": input_messages},
                config=config,
            )
            completion = codecs.decode(repr(response), "unicode_escape")
            completions.append(completion)
        # Save
        for index_in_curr_batch, output in enumerate(completions):
            generated_text = output
            index_in_data_list = index_global * batch_size + index_in_curr_batch
            item = test_dataset_list[index_in_data_list]
            item["completion"] = generated_text

    with open(output_path, 'w+t') as file:
        json.dump(test_dataset_list, file, indent=4)

    # deal with versions
    version_indexes = []
    if version == -1:
        version_indexes = all_version_indexes
    else:
        version_indexes = [version - 1]
    for index in version_indexes:
        if not os.path.exists(output_path.split(".js")[0] + f"_v{index + 1}.json"):
            with open(output_path.split(".js")[0] + f"_v{index + 1}.json", "w+t", encoding="utf-8") as file:  # also save different versions
                item_copy = [item for item in test_dataset_list if item['version'] == index + 1]
                json.dump(item_copy, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        help="The path of model to evaluate.",
                        default="mcma-agent-v1")
    parser.add_argument('--subject',
                        type=str,
                        help="The subject to evaluate.",
                        default="Calculus-Single-Variable")
    parser.add_argument('--version',
                        type=int,
                        help="Random version to evaluate. -1 for all versions.",
                        default=1)
    parser.add_argument('--prompt',
                        type=str,
                        help="The prompt template to use in evaluation.",
                        default="raw")
    parser.add_argument('--output_dir', type=str, default=os.path.join(proj_root_dir, ".results"))
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--test_count', type=int, default=1)
    parser.add_argument('--random_seed', default=None)

    args = parser.parse_args()
    out_dir = os.path.join(args.output_dir, args.model.split("/")[-1])
    mkdir(out_dir)
    subjects = []
    if args.subject == "all":
        subjects = SUBJECTS
    else:
        subjects = [args.subject]

    for subject in subjects:
        out_fn = os.path.join(out_dir, args.subject + ".json")
        if not os.path.exists(out_fn):
            print(f"Start evaluating {args.model.split('/')[-1]} on {subject} version {args.version}!")
            generate(
                model_path=args.model,
                dataset_path=os.path.join(proj_root_dir, ".data", subject + ".json"),
                version=args.version,
                output_path=out_fn,
                prompt=args.prompt,
                nproc=args.nproc,
                test_count=args.test_count,
                random_seed = args.random_seed,
            )
        else:
            print(f"{args.model.split('/')[-1]} on {subject} already exist!")
