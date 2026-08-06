# Importation rules.
# 1st, import standard library modules.
# 2nd, import 3rd party library modules.
# 3rd, import project-local modules.
# 4th, violate the above order only when violations are unavoidable.

import json
import os
import re
from typing import Any, Callable

import chardet
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from sympy import (
    E,
    FiniteSet,
    I,
    Intersection,
    Interval,
    Matrix,
    N,
    Union,
    pi,
    simplify,
    sqrt,
)
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.exceptions import SymPyDeprecationWarning
from tqdm import tqdm

# Declaration rules.
# 1st, declare local aliases.
# 2nd, declare private module attributes.
# 3rd, declare public module attributes.
# 4th, violate the above order only when violations are unavoidable.

proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
"""Project root directory."""

STRIP_STRS = [
    ":",
    ".",
    "/",
    ",",
    "#",
    "?",
    "$",
    '"',
    "'",
    # "ки" is the delimeter for Math-Shepherd
    "к",
    "и",
    # LaTeX
    "\\(",
    "\\)",
    "\\[",
    "\\]",
]
NO_TRAILING_STRS = ["(", "[", "{", "\\"] + STRIP_STRS
NO_PRECEDING_PUNCS = ["!", ")", "]", "}", "\\\\"] + STRIP_STRS
# Answer prefixes
PRM800K_ANS_PRRFIX = "# Answer"
GSM8K_ANS_PREFIX = "####"


def get_encoding_type(file_path):
    with open(file_path, 'rb') as f:
        sample = f.read(1024)
        cur_encoding = chardet.detect(sample)['encoding']
        return cur_encoding


def read_json(file_path):
    with open(file_path, 'r', encoding=get_encoding_type(file_path), errors="ignore") as f:
        data = json.load(f)
        return data


def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8', errors="ignore") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save2jsonl(name, data):
    with open(name, "w") as file:
        for dict_obj in data:
            json_str = json.dumps(dict_obj)
            file.write(json_str + "\n")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def readjsonl2list(name):
    data = []  # Create an empty list to store the dictionaries

    with open(name, "r") as file:
        for line in file:
            dict_obj = json.loads(line)
            data.append(dict_obj)
    return data


def contains_chinese(d):
    def is_chinese_char(ch):
        return '\u4e00' <= ch <= '\u9fff'

    def check(value):
        if isinstance(value, str):
            return any(is_chinese_char(ch) for ch in value)
        elif isinstance(value, dict):
            return any(check(v) for v in value.values())
        elif isinstance(value, list):
            return any(check(item) for item in value)
        return False

    return check(d)


def extract_code(text):
    pattern = r"```(?:python)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def norm_str2bool(s: str) -> bool | None:
    """Converts a string representation of a boolean value to its corresponding boolean value."""
    # TODO: deal with OL with boolean
    if s in ['T', 'Y']:
        return True
    elif s in ['F', "N"]:
        return False
    s = str(s).lower().strip().replace("noindent", "").split(" ")
    if any(pos in s for pos in ["yes", "true"]):
        return True
    elif any(neg in s for neg in ["no", "false"]):
        return False
    else:
        return None


def latex2sympy_fix(s: str):
    sp_symbol = parse_latex(s)

    if "," in s:
        first_term = None
        try:
            first_term = parse_latex(s.split(",")[0])
        except Exception:
            pass
        if sp_symbol == first_term:
            raise LaTeXParsingError(f"{s} != {first_term}")

    return sp_symbol


def latex2sympy_interval(s: str):
    """Parse LaTeX expression like (-\\infty,0] as SymPy Interval object."""
    s = s.replace(" ", "")

    if "\\cup" in s:
        exps = s.split("\\cup")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Union(*intervals)

    if "\\cap" in s:
        exps = s.split("\\cap")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Intersection(*intervals)

    if s.startswith("\\{") and s.endswith("\\}"):
        return FiniteSet(simplify(latex2sympy_fix(s[2:-2])))
    elif s.startswith("{") and s.endswith("}"):
        return FiniteSet(simplify(latex2sympy_fix(s[1:-1])))

    if s.startswith("("):
        left_open = True
        s = s[1:]
    elif s.startswith("\\("):
        left_open = True
        s = s[2:]
    elif s.startswith("["):
        left_open = False
        s = s[1:]
    elif s.startswith("\\["):
        left_open = False
        s = s[2:]
    else:
        raise ValueError(f"Invalid interval: {s}")

    if s.endswith(")"):
        right_open = True
        s = s[:-1]
    elif s.endswith("\\)"):
        right_open = True
        s = s[:-2]
    elif s.endswith("]"):
        right_open = False
        s = s[:-1]
    elif s.endswith("\\]"):
        right_open = False
        s = s[:-2]
    else:
        raise ValueError(f"Invalid interval: {s}")

    left, right = s.split(",")
    left = simplify(latex2sympy_fix(left))
    right = simplify(latex2sympy_fix(right))
    if left.is_comparable and right.is_comparable and left >= right:
        raise ValueError(f"Invalid interval: {left}, {right}")
    interval = Interval(left, right, left_open, right_open)

    return interval


PAREN_MAP = {
    r"\(": r"\)",
    r"\[": r"\]",
    r"\{": r"\}",
    "(": ")",
    "[": "]",
    "{": "}",
}

DATETIME_FMTS = [
    # Date formats
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    # Date and time formats
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M",
    # Time formats only
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
    "%I:%M %p",  # 24-hour and 12-hour formats
]

BASIC_FN_NAMES = (
    "sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|coth|sech|csch|log|ln|exp|arcsin|arccos|arctan|arcsec|arccsc|arccot|arcsinh|arccosh|arctanh|arcsech|arccsch|arccoth"
).split("|")

UNITS = [
    "hour",
    "minute",
    "min",
    "sec",
    "s",
    "second",
    "day",
    "week",
    "month",
    "year",
    "meter",
    "mile",
    "kg",
    "mg",
    "g",
    "t",
    "ton",
    "nm",
    "pm",
    "um",
    "μm",
    "m",
    "cm",
    "mm",
    "dm",
    "km",
    "kilometer",
    "inch",
    "feet",
    "ft",
    "piece",
    "bit",
    "hz",
    "Hz",
    "m/s",
    "km/s",
    "m/(min^2)",
    "billion",
    "eV",
    "V",
    "C",
    "s",
    "rad",
    "rad/min",
    "in",
    "cm^3",
    "V/h",
    "m^2",
    "L/min",
    "mi/hr",
    "lb",
    r"a\.?m\.?",
    r"(?<!\\)p\.?m\.?",  # 1\pm\sqrt{5}
]


def has_non_ascii(s):
    for char in s:
        if ord(char) > 127:
            return True
    return False


def is_querying4set(query):
    return "ind the" in query or ("all" in query and "separate" in query)


NDAYS_PER_WEEK = 7
WEEKDAY_ABBRS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAY_FULLS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def norm_str2weekday(s: str) -> str | None:
    """Converts a string representation of a weekday to its normalized form. Returns `None` if the input is not a valid weekday"""
    s = str(s).lower().strip()
    if " " in s:  # not a word
        return None

    for i_day in range(NDAYS_PER_WEEK):
        if s.startswith(WEEKDAY_ABBRS[i_day]):
            return WEEKDAY_FULLS[i_day].capitalize()
    return None


def parse(parser: Callable, s_to_parse: str, parse_errs: list[Exception]) -> Any | None:
    try:
        return parser(s_to_parse)
    except Exception as e:
        parse_errs.append(e)
    return None


def norm_deg(s: str) -> str:
    """Normalize expressions including degrees, except independent <num>\\circ"""
    s = s.replace("rad", "")
    s = re.sub(r"^(\d+) ?\^?\\?circ$", r"\1", s)
    s = re.sub(r"(\d+) ?\^?\\?circ", r"{\1*\\frac{\\pi}{180}}", s)

    return s


def is_set(s: str):
    return (
        re.search(r"[^a-z]or(x|[^a-z])", s) is not None
        or (s.startswith("{") and s.endswith("}"))
        or (s.startswith("\\{") and s.endswith("\\}"))
    )


def fix_sqrt(
    s: str,
) -> str:
    """Fixes the formatting of square root expressions in a given string."""
    _s = re.sub(r"\\?sqrt[\(\{\[](\w+)[\)\}\]]", r"\\sqrt{\1}", s)
    _s = re.sub(r"\\?sqrt\s*(\d+)", r"\\sqrt{\1}", _s)
    return _s


def fix_fracs(s: str) -> str:
    """Fixes the formatting of fractions in a given string."""
    substrs = s.split("\\frac")
    _s = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            _s += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                _s += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return s
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        _s += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        _s += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        _s += "{" + a + "}" + b + post_substr
                    else:
                        _s += "{" + a + "}" + b
    return _s


def fix_a_slash_b(s: str) -> str:
    """
    Fixes the formatting of fractions in a given string using regular expressions.
    """
    # Define a regular expression to match fractions. Here we match two parts: the numerator (a) and the denominator (b).
    # The numerator and denominator can be numbers (\d+) or expressions containing sqrt (sqrt\(.*?\)).
    # TODO: deal with 1.124/ 2.123
    fraction_pattern = r"(\b\d+\..*|sqrt\(.*?\))\/(\d+\..*|sqrt\(.*?\)\b)"

    # Use `re.sub` to replace the matched fractions with properly formatted fractions.
    result = re.sub(
        fraction_pattern, lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}", s
    )

    return result


def fix_inv_func(s: str) -> str:
    func_list = "arcsin|arccos|arctan|arcsec|arccsc|arccot|arcsinh|arccosh|arctanh|arcsech|arccsch|arccoth".split("|")
    conv_list = "asin|acos|atan|asec|acsc|acot|asinh|acosh|atanh|asech|acsch|acoth".split("|")
    for c, f in zip(conv_list, func_list):
        s = s.replace(c, f)
    return s


STR2NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def rm_latex_env(s: str, env: str) -> str:
    """Remove LaTeX environment from a string.

    Parameters
    ----------
    s : str
        The input string.
    env : str
        The LaTeX environment name to remove.

    Returns
    -------
    str
        The string with the specified LaTeX environment removed.
    """
    s = s.replace(f"\\begin{{{env}}}", "")
    s = s.replace(f"\\end{{{env}}}", "")
    return s


LATEX_CMDS = [
    "\\textbf",
    "\\textit",
    "\\textsl",
    "\\texttt",
    "\\textsc",
    "\\textsf",
    "\\textrm",
    "\\mathrm",
    "\\mathbf",
    "\\mathit",
    "\\mathsf",
    "\\mathtt",
    "\\mathbb",
    "\\mathcal",
    "\\mathscr",
    "\\mathfrak",
    "\\bm",
    "\\em",
    "\\emph",
    "\\underline",
    "\\overline",
    "\\tiny",
    "\\scriptsize",
    "\\footnotesize",
    "\\small",
    "\\normalsize",
    "\\large",
    "\\Large",
    "\\LARGE",
    "\\huge",
    "\\Huge",
    "\\newline",
    "\\par",
    "\\noindent",
    "\\indent",
    "\\footnote",
    "\\cite",
    "\\ref",
    "\\label",
    "\\textsuperscript",
    "\\textsubscript",
    "\\text",
    "\\mbox",
    "\\renewcommand{\\arraystretch}",
]

LATEX_FMT_ENVS = [
    # Align
    "align",
    "align*",
    "center",
    "flushleft",
    "flushright",
]
LATEX_LIST_ENVS = [
    "itemize",
    "enumerate",
    "description",
]


SIMPLE_RM_STRS = [
    "\n",
    "\t",
    "approximately",
    "'",
    '"',
    "\\$",
    "$",
    "￥",
    "£",
    "€",
    "{,}",
    "\\!",
    "\\,",
    "\\:",
    "\\;",
    "\\quad",
    "\\qquad",
    "\\space",
    "\\thinspace",
    "\\medspace",
    "\\thickspace",
    "~,",
    "\\ ",
    # Note the order
    "\\\\%",
    "\\%",
    "%",
    "\\left",
    "\\right",
    "^{\\circ}",
    "^\\circ",
]

SIMPLE_REPLACE_MAP = {
    "∪": "\\cup",
    "U": "\\cup",
    "π": "\\pi",
    "∞": "\\infty",
    "∈": "\\in",
    "∩": "\\cap",
    "−": "-",
    "\\item": ",",
    "and": ",",
    ";": ",",
    "infinity": "\\infty",
    "+\\infty": "\\infty",
    "tfrac": "frac",
    "dfrac": "frac",
    "\\approx": "=",
    "\\times": "*",
    "\\cdot": "*",
    "{.": "{0.",  # "{0." equivalent to "{."
    " .": " 0.",  # " 0." equivalent to " ."
    ":": "/",  # Ratio like 3:2
}


def make_prompt(prompt_dict, data):
    prompt_prefix_multiple = (
        "The following is an undergraduate-level mathematical problem in {subject}. You need to solve the problem by completing all placeholders [ANS].\n\n"
        "This problem involves {num_of_answers} placeholders [ANS] to be completed. Their answer types are, in order, {answer_type_description}.\n\n"
        "Problem:\n{problem}\n\n"
        'All mathematical formulas and symbols you output should be represented with LaTeX. Please end your response with: "The final answers are \\boxed{ANSWER}", where ANSWER should be the sequence of your final answers, separated by commas.'
    )

    prompt_prefix_single = (
        "The following is an undergraduate-level mathematical problem in {subject}. You need to solve the problem by completing all placeholders [ANS].\n\n"
        "This problem involves only one placeholders [ANS] to be completed. The answer type is {answer_type_description}.\n\n"
        "Problem:\n{problem}\n\n"
        'All mathematical formulas and symbols you output should be represented with LaTeX. Please end your response with: "The final answer is \\boxed{ANSWER}", where ANSWER should be your final answer.'
    )

    type2descriptions = {
        "UOL": 'an unordered list of answers surrounded by parentheses with any answer types, for example, (1, x^2, True), where "unordered list" means changing the order of elements results in the same answer',
        "OL": 'an ordered list of answers surrounded by parentheses with any answer types, for example, (1, x^2, True), where "ordered list" means changing the order of elements results in different answers',
        "INT": 'a range inteval',
        "TF": 'either True or False',
        "EX": 'an expression',
        "EQ": 'an equation',
        "MCS": "one option of a multiple choice question with options {options}",
        "MCM": "more than one option concatenated without space or commas of a multiple choice question with options {options}, for example: BD",
        "NV": "a numerical value without units",
        "OE": "a word, phrase, term or string that satisfies the requirements of the problem"
    }

    for item in data:
        item['prompt'] = prompt_dict['sys_prompt'] + prompt_dict['query_prompt']
        if len(item['answer']) == 1:
            desc = type2descriptions[item['answer_type'][0]]
            if item['answer_type'][0] in ['MCS', 'MCM']:
                desc = desc.format(options=item['options'][0])
            item['prompt'] += prompt_prefix_single.format(subject=item['subject'],
                                                          answer_type_description=desc, problem=item['problem'], ANSWER="{ANSWER}")
        else:
            desc = ""
            for i, ty in enumerate(item['answer_type']):
                if i == 0:
                    desc += type2descriptions[ty]
                else:
                    desc += ", " + type2descriptions[ty]
                if ty in ['MCS', 'MCM']:
                    desc = desc.format(options=item['options'][i])
            item['prompt'] += prompt_prefix_multiple.format(subject=item['subject'], num_of_answers=len(
                item['answer']), answer_type_description=desc, problem=item['problem'], ANSWER="{ANSWER}")
        item['prompt'] += prompt_dict['prompt_after_query'] + \
            prompt_dict['resp_prompt'] + prompt_dict['prompt_before_resp']
    return data


def make_prompt_with_template(tokenizer, data, sys_prompt=""):
    prompt_prefix_multiple = (
        "The following is an undergraduate-level mathematical problem in {subject}. You need to solve the problem by completing all placeholders [ANS].\n\n"
        "This problem involves {num_of_answers} placeholders [ANS] to be completed. Their answer types are, in order, {answer_type_description}.\n\n"
        "Problem:\n{problem}\n\n"
        'All mathematical formulas and symbols you output should be represented with LaTeX. Please end your response with: "The final answers are \\boxed{ANSWER}", where ANSWER should be the sequence of your final answers, separated by commas.'
    )

    prompt_prefix_single = (
        "The following is an undergraduate-level mathematical problem in {subject}. You need to solve the problem by completing all placeholders [ANS].\n\n"
        "This problem involves only one placeholders [ANS] to be completed. The answer type is {answer_type_description}.\n\n"
        "Problem:\n{problem}\n\n"
        'All mathematical formulas and symbols you output should be represented with LaTeX. Please end your response with: "The final answer is \\boxed{ANSWER}", where ANSWER should be your final answer.'
    )

    type2descriptions = {
        "UOL": 'an unordered list of answers surrounded by parentheses with any answer types, for example, (1, x^2, True), where "unordered list" means changing the order of elements results in the same answer',
        "OL": 'an ordered list of answers surrounded by parentheses with any answer types, for example, (1, x^2, True), where "ordered list" means changing the order of elements results in different answers',
        "INT": 'a range inteval',
        "TF": 'either True or False',
        "EX": 'an expression',
        "EQ": 'an equation',
        "MCS": "one option of a multiple choice question with options {options}",
        "MCM": "more than one option concatenated without space or commas of a multiple choice question with options {options}, for example: BD",
        "NV": "a numerical value without units",
        "OE": "a word, phrase, term or string that satisfies the requirements of the problem"
    }

    for item in data:
        # item['prompt'] = prompt_dict['sys_prompt'] + prompt_dict['query_prompt']
        item['prompt'] = ""
        if len(item['answer']) == 1:
            desc = type2descriptions[item['answer_type'][0]]
            if item['answer_type'][0] in ['MCS', 'MCM']:
                desc = desc.format(options=item['options'][0])
            item['prompt'] += prompt_prefix_single.format(subject=item['subject'],
                                                          answer_type_description=desc, problem=item['problem'], ANSWER="{ANSWER}")
        else:
            desc = ""
            for i, ty in enumerate(item['answer_type']):
                if i == 0:
                    desc += type2descriptions[ty]
                else:
                    desc += ", " + type2descriptions[ty]
                if ty in ['MCS', 'MCM']:
                    desc = desc.format(options=item['options'][i])
            item['prompt'] += prompt_prefix_multiple.format(subject=item['subject'], num_of_answers=len(
                item['answer']), answer_type_description=desc, problem=item['problem'], ANSWER="{ANSWER}")
        # item['prompt'] += prompt_dict['prompt_after_query'] + prompt_dict['resp_prompt'] + prompt_dict['prompt_before_resp']
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": item['prompt']}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        item['prompt'] = text
    return data


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
