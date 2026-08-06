# Importation rules.
# 1st, import standard library modules.
# 2nd, import 3rd party library modules.
# 3rd, import project-local modules.
# 4th, violate the above order only when violations are unavoidable.

import argparse
import os.path
import random
import time
from collections import defaultdict
from multiprocessing import Lock, Manager, Pipe, Process
from typing import Optional

import numpy as np
from tqdm import tqdm

from ugmb_vlyc.generate import SUBJECTS, VERSION_COUNT, random_seed
from ugmb_vlyc.judge_marj import Judger
from ugmb_vlyc.utils import *

# Declaration rules.
# 1st, declare local aliases.
# 2nd, declare private module attributes.
# 3rd, declare public module attributes.
# 4th, violate the above order only when violations are unavoidable.

random.seed(random_seed)
ugmath_judger = Judger(strict_extract=True)

def eval_for_single_item(data_item, precision=1e-8):
    """
    Check the correctness for single item. 

    Parameters:
        data_item: Data to be checked. 
            data format:
            {
                "id": "Trigonometry_0072",
                "subject": "Trigonometry",
                "topic": "Trigonometric functions",
                "subtopic": "Trigonometric functions of special angles",
                "level": "2",
                "keywords": [
                    "tangent",
                    "cotangent",
                    "cosecant",
                    "secant"
                ],
                "problem": "Questions 8-16:\nFind the exact value of each without using a calculator:\na) $\\tan{\\left(\\frac{2 \\pi}{3} \\right)}$=[ANS]\nb) $\\tan{\\left(\\frac{5 \\pi}{4} \\right)}$=[ANS]\nc) $\\cot{\\left(\\frac{\\pi}{3} \\right)}$=[ANS]\nd) $\\sec{\\left(\\frac{7 \\pi}{6} \\right)}$=[ANS]\ne) $\\csc{\\left(\\frac{\\pi}{4} \\right)}$=[ANS]",
                "answer": [
                    "-sqrt(3)",
                    "1",
                    "1/sqrt(3)",
                    "-2/sqrt(3)",
                    "sqrt(2)"
                ],
                "version": 3,
                "answer_type": [
                    "NV",
                    "NV",
                    "NV",
                    "NV",
                    "NV"
                ],
                "options": [
                    [],
                    [],
                    [],
                    [],
                    []
                ],
                "prompt": "User: The following is an undergraduate-level mathematical problems in Trigonometry. You need to solve the problem by completing all placeholders [ANS].\n\nThis problem involves 5 placeholders [ANS] to be completed. Their answer types are, in order, a numerical value, a numerical value, a numerical value, a numerical value, a numerical value.\n\nProblem:\nQuestions 8-16:\nFind the exact value of each without using a calculator:\na) $\\tan{\\left(\\frac{2 \\pi}{3} \\right)}$=[ANS]\nb) $\\tan{\\left(\\frac{5 \\pi}{4} \\right)}$=[ANS]\nc) $\\cot{\\left(\\frac{\\pi}{3} \\right)}$=[ANS]\nd) $\\sec{\\left(\\frac{7 \\pi}{6} \\right)}$=[ANS]\ne) $\\csc{\\left(\\frac{\\pi}{4} \\right)}$=[ANS]\n\nAll mathematical formulas and symbols you output should be represented with LaTeX. Please end your response with: \"The final answers are \\boxed{ANSWER}\", where ANSWER should be the sequence of your final answers, separated by commas.\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant: ",
                "completion": "\na) $\\tan{\\left(\\frac{2 \\pi}{3} \\right)}$\nSince $\\frac{2 \\pi}{3}$ is in the second quadrant, we know that $\\tan{\\left(\\frac{2 \\pi}{3} \\right)}$ is positive. Also, $\\frac{2 \\pi}{3}$ is $\\frac{\\pi}{3}$ less than $\\pi$, so $\\tan{\\left(\\frac{2 \\pi}{3} \\right)} = -\\tan{\\left(\\frac{\\pi}{3} \\right)}$. Since $\\tan{\\left(\\frac{\\pi}{3} \\right)} = \\sqrt{3}$, we have $\\tan{\\left(\\frac{2 \\pi}{3} \\right)} = -\\sqrt{3}$.\n\nb) $\\tan{\\left(\\frac{5 \\pi}{4} \\right)}$\nSince $\\frac{5 \\pi}{4}$ is in the third quadrant, we know that $\\tan{\\left(\\frac{5 \\pi}{4} \\right)}$ is negative. Also, $\\frac{5 \\pi}{4}$ is $\\frac{\\pi}{4}$ less than $\\pi$, so $\\tan{\\left(\\frac{5 \\pi}{4} \\right)} = -\\tan{\\left(\\frac{\\pi}{4} \\right)}$. Since $\\tan{\\left(\\frac{\\pi}{4} \\right)} = 1$, we have $\\tan{\\left(\\frac{5 \\pi}{4} \\right)} = -1$.\n\nc) $\\cot{\\left(\\frac{\\pi}{3} \\right)}$\nSince $\\frac{\\pi}{3}$ is in the first quadrant, we know that $\\cot{\\left(\\frac{\\pi}{3} \\right)}$ is positive. Also, $\\frac{\\pi}{3}$ is $\\frac{\\pi}{6}$ less than $\\frac{\\pi}{2}$, so $\\cot{\\left(\\frac{\\pi}{3} \\right)} = \\tan{\\left(\\frac{\\pi}{6} \\right)}$. Since $\\tan{\\left(\\frac{\\pi}{6} \\right)} = \\frac{1}{\\sqrt{3}}$, we have $\\cot{\\left(\\frac{\\pi}{3} \\right)} = \\frac{\\sqrt{3}}{1} = \\sqrt{3}$.\n\nd) $\\sec{\\left(\\frac{7 \\pi}{6} \\right)}$\nSince $\\frac{7 \\pi}{6}$ is in the fourth quadrant, we know that $\\sec{\\left(\\frac{7 \\pi}{6} \\right)}$ is positive. Also, $\\frac{7 \\pi}{6}$ is $\\frac{\\pi}{6}$ more than $\\pi$, so $\\sec{\\left(\\frac{7 \\pi}{6} \\right)} = \\sec{\\left(\\frac{\\pi}{6} \\right)}$. Since $\\sec{\\left(\\frac{\\pi}{6} \\right)} = \\frac{1}{\\cos{\\left(\\frac{\\pi}{6} \\right)}}$, and $\\cos{\\left(\\frac{\\pi}{6} \\right)} = \\frac{\\sqrt{3}}{2}$, we have $\\sec{\\left(\\frac{7 \\pi}{6} \\right)} = \\frac{2}{\\sqrt{3}} = \\frac{2\\sqrt{3}}{3}$.\n\ne) $\\csc{\\left(\\frac{\\pi}{4} \\right)}$\nSince $\\frac{\\pi}{4}$ is in the first quadrant, we know that $\\csc{\\left(\\frac{\\pi}{4} \\right)}$ is positive. Also, $\\frac{\\pi}{4}$ is $\\frac{\\pi}{4}$ less than $\\frac{\\pi}{2}$, so $\\csc{\\left(\\frac{\\pi}{4} \\right)} = \\sec{\\left(\\frac{\\pi}{4} \\right)}$. Since $\\sec{\\left(\\frac{\\pi}{4} \\right)} = \\frac{1}{\\cos{\\left(\\frac{\\pi}{4} \\right)}}$, and $\\cos{\\left(\\frac{\\pi}{4} \\right)} = \\frac{\\sqrt{2}}{2}$, we have $\\csc{\\left(\\frac{\\pi}{4} \\right)} = \\frac{1}{\\frac{\\sqrt{2}}{2}} = \\sqrt{2}$.\n\nThe final answers are $\\boxed{-\\sqrt{3}, -1, \\sqrt{3}, \\frac{2\\sqrt{3}}{3}, \\sqrt{2}}$."
            }

    Returns:
        correctness: A list that contains the correctness for all prompts. 
    """
    steps = data_item['completion']
    correctness = ugmath_judger.judge(data_item['completion'],
                                      data_item['answer'],
                                      data_item['answer_type'],
                                      data_item['options'],
                                      precision)
    gt = [ugmath_judger.norm_ans_str(item, at) for item, at in zip(data_item['answer'], data_item['answer_type'])]
    extracted_pred = ugmath_judger.extract_ans(data_item['completion'])
    extracted_pred = ugmath_judger.split_by_comma(extracted_pred)
    extracted_pred = [ugmath_judger.norm_ans_str(item, at)
                      for item, at in zip(extracted_pred, data_item['answer_type'])]
    flag = False
    for item in data_item['answer_type']:
        if item != "NV":
            flag = True
    if not flag:
        for item in data_item['answer']:
            try:
                float(item)
            except:
                flag = True
                continue
    if not correctness and flag:
        correctness, msg = ugmath_judger.aux_judge(data_item['completion'],
                                                   data_item['answer'],
                                                   data_item['problem'])
    else:
        msg = None
    return correctness, gt, extracted_pred, msg


def map_result_to_data(data_list, check_result_list):
    """
    Combine the data_list and check_result_list together. 
    Add the check result to the items in data list. 

    Parameters:
    - check_result_list: A list that contains the correctness of the original data. 
        data format:
        {
            "index": 0, 
            "data": [
                True
            ]
        }

    Returns:
    - data_list: A list that contains the original data with the check result. 
        data format:
        {
            "question": "Joan found 70 seashells on the beach . she gave Sam some of her seashells . She has 27 seashell . How many seashells did she give to Sam ? ",
            "type": "MAWPS",
            "subtype": "addsub",
            "answer": "43",
            "level": 0,
            "completion": [
                "\n\nTo find out how many seashells Joan gave to Sam, we need to subtract the number of seashells she has left from the total number of seashells she found.\n\nJoan found 70 seashells and has 27 seashells left.\n\nSo, we subtract 27 from 70:\n\n70 - 27 = 43\n\nJoan gave 43 seashells to Sam.\n\nThe answer is: 43."
            ],
            "correctness": [
                True
            ]
        }
    - counter_list: A list that containseval_functions.utils.counters for different prompts. 
    """

    for result_item in check_result_list:
        check_result = result_item["data"]
        data_item = data_list[result_item["index"]]
        data_item["normalized_gt"] = check_result[1]
        data_item["extraced_answer"] = check_result[2]
        data_item["correctness"] = check_result[0]
        data_item['model_judge_msg'] = check_result[-1]

    return data_list


def eval_process(conn, precision=1e-8):
    """
    Inner function for checking correctness with timeout. 
    Carry out the correctness check. 

    Parameters:
        conn: A multiprocessing.Pipe object for send the results back. 
        config: The policy and settings for evaluations. Please refer to eval/eval_config.json for more details. 

    Returns:
        None
    """
    conn.send("Ready")
    while True:
        data_item = conn.recv()
        result = eval_for_single_item(data_item, precision)
        conn.send(result)


def eval_with_timeout(counter, lock, data_path, conn, precision=1e-8):
    """
    Process for checking correctness with timeout. 
    We start a new child process to check and use the parent process to monitor. 


    Parameters:
        counter: A multiprocessing.Manage.Value('int') object for tracting progress. The value indicates the index of the item we shall process now. 
        lock:  A multiprocessing.Lock object for `counter`. 
        data_path: Path to the data to be evaluated. 
        config: The policy and settings for evaluations. Please refer to eval/eval_config.json for more details. 
        conn: A multiprocessing.Pipe object for send the results back. 

    Returns:
        None
    """

    # TODO: We need better IPC for big dict, loading data costs a lot of time.
    data_list = read_json(data_path)

    # Pipe for connection between this process and the process started within this process.
    parent_conn, child_conn = Pipe()

    # Start a new process to carry out the actual work.
    p = Process(target=eval_process, args=(child_conn, precision))
    p.start()
    # Wait for the child process to be ready.
    # The child process will send "Ready".
    parent_conn.recv()

    result_list = []

    while True:

        # Get current index
        with lock:
            curr_index = counter.value
            if curr_index >= len(data_list):
                break
            counter.value += 1

        # Send data to child process
        data_item = data_list[curr_index]
        parent_conn.send(data_item)

        # Wait the child 5 seconds
        if parent_conn.poll(600):
            return_data = parent_conn.recv()
            result_list.append(dict(index=curr_index, data=return_data))
        else:
            # If the child process does not return a result, restart the process
            result_list.append(dict(index=curr_index, data=(False, None, None)))

    # Send all the results back
    conn.send(result_list)

    # Close child process and pipe
    p.terminate()
    child_conn.close()
    parent_conn.close()


def progress_bar_update(counter, total):
    """
    Process for update progress bar. 

    Parameters:
    - counter: A multiprocessing.Manage.Value('int') object for tracting progress. 
    - total: The sum of all tasks. 

    Returns:
    None
    """
    pbar = tqdm(total=total)

    while counter.value < total:
        pbar.update(counter.value - pbar.n)
        time.sleep(0.1)

    pbar.update(counter.value - pbar.n)
    pbar.close()


def eval_file(data_path: str, save_path: str, precision: float = 1e-8, n_proc: int = 16):
    """
    Evaluate the generated response. 

    Parameters: 
        data_path: The path to the generated response. 
        save_path: The path to save the results. The default save_path is "evaluation/{data_file_name}"
    """
    data_list = read_json(data_path)
    if data_path != save_path:
        with Manager() as manager:
            lock = Lock()
            counter = manager.Value('i', 0)

            p_pbar = Process(target=progress_bar_update, args=(counter, len(data_list)))
            p_pbar.start()

            p_list = []
            pipe_list = []
            for _ in range(n_proc):
                parent_conn, child_conn = Pipe()
                pipe_list.append((parent_conn, child_conn))
                p = Process(target=eval_with_timeout, args=(counter, lock, data_path, child_conn, precision))
                p.start()
                p_list.append(p)

            result_list = []
            for p, (parent_conn, child_conn) in zip(p_list, pipe_list):
                result_list += parent_conn.recv()
                p.join()
                parent_conn.close()
                child_conn.close()
            p_pbar.join()

        data_list = map_result_to_data(data_list, result_list)
        with open(save_path, 'w') as f:
            json.dump(data_list, f, indent=4)

    # report metrics

    df = pd.DataFrame(data_list)

    # Function to aggregate calculations
    def calculate_metrics(group):
        correct_count = group['correctness'].sum()
        total_count = group['correctness'].count()
        accuracy = correct_count / total_count if total_count > 0 else 0
        return pd.Series({
            'accuracy': accuracy,
            'total': total_count,
            'true': correct_count
        })

    combined_metrics = df.groupby(['version', 'subject', 'level']).apply(calculate_metrics).reset_index()
    combined_metrics_version_sub = df.groupby(['version', 'subject']).apply(calculate_metrics).reset_index()
    combined_metrics_version = df.groupby(['version']).apply(calculate_metrics).reset_index()

    print("\nAcc for each Version, Subject, and Level combination:")
    print(combined_metrics.to_string(index=False))
    print("\nAcc for each Version, Subject combination:")
    print(combined_metrics_version_sub.to_string(index=False))
    print("\nAcc for each Version:")
    print(combined_metrics_version.to_string(index=False))

    # report eacc, aacc, cacc, robustness
    aacc = np.mean([item['correctness'] for item in data_list])
    # Step 1: Organize Data by Version
    version2data = {}
    for i in range(VERSION_COUNT):
        version_key = str(i + 1)
        version2data[version_key] = sorted(
            [item for item in data_list if item['version'] == i + 1],
            key=lambda x: x['id']
        )
        version2data[version_key] = [item['correctness'] for item in version2data[version_key]]

    # Step 2: Compute Counts
    id_to_version_correctness = defaultdict(list)

    # Collect correctness by id across all versions
    for item in data_list:
        id_to_version_correctness[item['id']].append(item['correctness'])

    # Calculate all true within versions and at least one true across versions
    all_true_within_versions_count = 0
    at_least_one_true_across_versions_count = 0

    all_true_within_versions = set()
    at_least_one_true = set()

    for item_id, correctness_list in id_to_version_correctness.items():
        if all(correctness_list):
            all_true_within_versions_count += 1
            all_true_within_versions.add(item_id)

        if any(correctness_list):
            at_least_one_true_across_versions_count += 1
            at_least_one_true.add(item_id)

    print(f'AAcc:\t{aacc: .4f}')
    print(f'EAcc:\t{all_true_within_versions_count / len(version2data[str(1)]): .4f}')
    print(f'CAcc:\t{at_least_one_true_across_versions_count / len(version2data[str(1)]): .4f}')
    print(f'Delta:\t{aacc - all_true_within_versions_count / len(version2data[str(1)]): .4f}')
    print(f'RE:\t{(aacc - all_true_within_versions_count / len(version2data[str(1)])) / (all_true_within_versions_count / len(version2data[str(1)])): .4f}')

    filename = save_path.split(".json")[0]+".txt"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("Acc for each Version, Subject, and Level combination:\n")
        file.write(combined_metrics.to_string(index=False))
        file.write("\nAcc for each Version, Subject combination:\n")
        file.write(combined_metrics_version_sub.to_string(index=False))
        file.write("\nAcc for each Version:\n")
        file.write(combined_metrics_version.to_string(index=False))
        file.write(f'\n\naacc: {aacc: .4f}\t')
        file.write(f'\teacc: {all_true_within_versions_count / len(version2data[str(1)]): .4f}')
        file.write(f'\tcacc: {at_least_one_true_across_versions_count / len(version2data[str(1)]): .4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        help="The path of model to evaluate.")
    parser.add_argument('--subject',
                        type=str,
                        help="The subject to evaluate.",
                        default="all")
    parser.add_argument('--output_dir', type=str, default=os.path.join(proj_root_dir, ".results"))
    parser.add_argument('--precision', type=float, default=1e-3)
    args = parser.parse_args()
    out_dir = os.path.join(args.output_dir, args.model_path.split("/")[-1])

    subjects = []
    if args.subject == "all":
        subjects = SUBJECTS
    else:
        subjects = [args.subject]

    for subject in subjects:
        input_path = os.path.join(out_dir, f"{subject}.json")
        output_path = os.path.join(out_dir, f"{subject}_eval_marj.json")
        if os.path.exists(input_path):
            eval_file(
                data_path=input_path,
                save_path=output_path,
                precision=args.precision
            )
            print(f"{subject} generated.")
        else:
            print(f"{input_path = } does not exist.")

    all_subjects_data = []
    for subject in subjects:
        input_path = os.path.join(out_dir, subject + "_eval_marj.json")
        if os.path.exists(input_path):
            all_subjects_data += read_json(input_path)
            print(f"{input_path = } data added.")
        else:
            print(f"{input_path = } does not exist.")
        print(f"{subject} results loaded!")
    input_path = os.path.join(out_dir, "all_subjects_eval_marj.json")
    with open(input_path, 'w+t', encoding="utf-8") as f:
        json.dump(all_subjects_data, f, indent=4)
    print(f"All subjects evaluation results saved to {input_path}")
    output_path = os.path.join(out_dir, "all_subjects_eval_marj.json")
    eval_file(
        data_path=input_path,
        save_path=output_path,
        precision=args.precision
    )
