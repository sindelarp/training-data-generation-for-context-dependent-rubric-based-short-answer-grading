import glob
import json
import itertools
import random
import re
import argparse
import os

import dspy
import tqdm

from dotenv import load_dotenv

from .dspy_signatures import (
    TextCleaner,
    QuestionGenerator,
    QuestionCritic,
    GenerateGradingRubric,
    GradingRubricCritic,
    GenerateCorrectAnswer,
    CorrectAnswerCritic,
    GenerateSemicorrectAnswer,
    SemicorrectAnswerCritic,
    GenerateIncorrectAnswer,
    IncorrectAnswerCritic,
)

def split_into_chunks(text: str, min_size: int = 150, max_size: int = 800):
    parts = re.split(
        r'(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bProf)(?<!\bRev)(?<!\bSt)(?<!\bMt)(?<!\bJr)(?<!\bSr)(?<!\bvs)(?<!\betc)(?<!\b[A-Z])([.!?]+[\'"”’\)\]}]*)\s+(?=[A-Z0-9])',
        text,
    )
    sentences = [
        (s + p).strip()
        for s, p in zip(parts[0::2], parts[1::2] + [""])
        if (s + p).strip()
    ]
    chunks = []
    current_chunk = ""
    target_length = random.randint(min_size, max_size)
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) + 1 <= target_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            target_length = random.randint(min_size, max_size)
    if current_chunk and len(current_chunk.split()) >= min_size:
        chunks.append(current_chunk.strip())
    return chunks

def main(args, api_key):
    random.seed(42)

    contexts = []

    for file in itertools.chain.from_iterable(glob.glob(g) for g in args.data_globs):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = split_into_chunks(text)
        for chunk in chunks:
            contexts.append(
                {"fileid": "devset/" + file.replace("\\", "/"), "context": chunk}
            )

    with open("all_contexts.json", "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False, indent=4)

    lm = dspy.LM(
        "openai/gpt-oss-120b",
        base_url=args.host_url,
        api_key=api_key,
        max_tokens=16000,
    )
    dspy.configure(lm=lm)

    for context in tqdm.tqdm(contexts):
        cleaned_text = dspy.Predict(TextCleaner)(text=context["context"]).cleaned_text
        context["context"] = cleaned_text

    with open("all_contexts_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False, indent=4)

    for i, context in enumerate(tqdm.tqdm(contexts)):
        generated_questions = dspy.Predict(QuestionGenerator)(context=context["context"]).questions
        critiques = dspy.Predict(QuestionCritic)(context=context["context"], questions=generated_questions).critiques
        context["generated_questions"] = generated_questions
        context["critiques"] = critiques
        questions = [q for q, c in zip(generated_questions, critiques) if c]
        context["questions"] = questions

    with open("all_contexts_with_questions.json", "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False, indent=4)

    flat_context_questions = []
    for context in contexts:
        for question in context["questions"]:
            flat_context_questions.append(
                {
                    "fileid": context["fileid"],
                    "context": context["context"],
                    "question": question,
                }
            )

    with open("all_contexts_with_questions_flat.json", "w", encoding="utf-8") as f:
        json.dump(flat_context_questions, f, ensure_ascii=False, indent=4)

    pbar = tqdm.tqdm(flat_context_questions)
    for i, item in enumerate(pbar):
        attempt = 1
        correct_rubric = False
        critique = None
        full_credit = None
        partial_credit = None
        no_credit = None
        while not correct_rubric and attempt <= 10:
            pbar.set_postfix_str(f"Attempt: {attempt}")
            generated_rubric = dspy.Predict(GenerateGradingRubric)(
                context=item["context"],
                question=item["question"],
                full_credit_example=full_credit,
                partial_credit_example=partial_credit,
                no_credit_example=no_credit,
                example_critique=critique,
            )
            full_credit = generated_rubric.full_credit
            partial_credit = generated_rubric.partial_credit
            no_credit = generated_rubric.no_credit
            rubric_critique = dspy.Predict(GradingRubricCritic)(
                context=item["context"],
                question=item["question"],
                full_credit=full_credit,
                partial_credit=partial_credit,
                no_credit=no_credit,
            )
            correct_rubric = rubric_critique.follows_criteria
            critique = rubric_critique.critique
            attempt += 1
        if correct_rubric:
            item["rubrics"] = {
                "FC": full_credit,
                "PC": partial_credit,
                "NC": no_credit,
            }
        else:
            print(f"Failed to generate a correct rubric for item {i}")

    with open("all_contexts_questions_rubrics.json", "w", encoding="utf-8") as f:
        json.dump(
            [item for item in flat_context_questions if "rubrics" in item],
            f,
            indent=4,
            ensure_ascii=False,
        )

    with open("all_contexts_questions_rubrics.json", "r", encoding="utf-8") as f:
        flat_context_questions = json.load(f)

    for i, item in enumerate(tqdm.tqdm(flat_context_questions)):
        generated_answers = dspy.Predict(GenerateCorrectAnswer)(
            context=item["context"],
            question=item["question"],
            grading_criteria=item["rubrics"]["FC"],
        ).answers
        critiques = dspy.Predict(CorrectAnswerCritic)(
            context=item["context"],
            question=item["question"],
            grading_criteria=item["rubrics"]["FC"],
            answers=generated_answers
        ).critiques
        item["generated_answers"] = generated_answers
        item["critiques"] = critiques
        answers = [q for q, c in zip(generated_answers, critiques) if c]
        item["answers"] = answers

    with open("all_contexts_questions_rubrics_correct_answers.json", "w", encoding="utf-8") as f:
        json.dump(flat_context_questions, f, ensure_ascii=False, indent=4)

    with open("all_contexts_questions_rubrics.json", "r", encoding="utf-8") as f:
        flat_context_questions = json.load(f)

    for i, item in enumerate(tqdm.tqdm(flat_context_questions)):
        generated_answers = dspy.Predict(GenerateSemicorrectAnswer)(
            context=item["context"],
            question=item["question"],
            grading_criteria=item["rubrics"]["PC"],
        ).answers
        critiques = dspy.Predict(SemicorrectAnswerCritic)(
            context=item["context"],
            question=item["question"],
            grading_criteria=item["rubrics"]["PC"],
            answers=generated_answers,
        ).critiques
        item["generated_answers"] = generated_answers
        item["critiques"] = critiques
        answers = [q for q, c in zip(generated_answers, critiques) if c]
        item["answers"] = answers

    with open(
        "all_contexts_questions_rubrics_semicorrect_answers.json", "w", encoding="utf-8"
    ) as f:
        json.dump(flat_context_questions, f, ensure_ascii=False, indent=4)

    with open("all_contexts_questions_rubrics.json", "r", encoding="utf-8") as f:
        flat_context_questions = json.load(f)

    for i, item in enumerate(tqdm.tqdm(flat_context_questions)):
        generated_answers = dspy.Predict(GenerateIncorrectAnswer)(
            context=item["context"],
            question=item["question"],
            grading_criteria=item["rubrics"]["NC"],
        ).answers
        critiques = dspy.Predict(IncorrectAnswerCritic)(
            context=item["context"],
            question=item["question"],
            grading_criteria=item["rubrics"]["NC"],
            answers=generated_answers,
        ).critiques
        item["generated_answers"] = generated_answers
        item["critiques"] = critiques
        answers = [q for q, c in zip(generated_answers, critiques) if c]
        item["answers"] = answers

    with open(
        "all_contexts_questions_rubrics_incorrect_answers.json", "w", encoding="utf-8"
    ) as f:
        json.dump(flat_context_questions, f, ensure_ascii=False, indent=4)

    all_data = []

    with open(
        "all_contexts_questions_rubrics_correct_answers.json", "r", encoding="utf-8"
    ) as f:
        correct_answers = json.load(f)

    with open(
        "all_contexts_questions_rubrics_semicorrect_answers.json", "r", encoding="utf-8"
    ) as f:
        semicorrect_answers = json.load(f)

    with open(
        "all_contexts_questions_rubrics_incorrect_answers.json", "r", encoding="utf-8"
    ) as f:
        incorrect_answers = json.load(f)

    for item in correct_answers:
        new_item = {}
        new_item["label"] = 2
        new_item["lang"] = "en"
        new_item["fileid"] = item["fileid"]
        for answer in item["answers"]:
            new_item_copy = new_item.copy()
            new_item_copy["input"] = {}
            new_item_copy["input"]["context"] = item["context"]
            new_item_copy["input"]["question"] = item["question"]
            new_item_copy["input"]["rubrics"] = item["rubrics"]
            new_item_copy["input"]["answer"] = answer
            all_data.append(new_item_copy)

    for item in semicorrect_answers:
        new_item = {}
        new_item["label"] = 1
        new_item["lang"] = "en"
        new_item["fileid"] = item["fileid"]
        for answer in item["answers"]:
            new_item_copy = new_item.copy()
            new_item_copy["input"] = {}
            new_item_copy["input"]["context"] = item["context"]
            new_item_copy["input"]["question"] = item["question"]
            new_item_copy["input"]["rubrics"] = item["rubrics"]
            new_item_copy["input"]["answer"] = answer
            all_data.append(new_item_copy)

    for item in incorrect_answers:
        new_item = {}
        new_item["label"] = 0
        new_item["lang"] = "en"
        new_item["fileid"] = item["fileid"]
        for answer in item["answers"]:
            new_item_copy = new_item.copy()
            new_item_copy["input"] = {}
            new_item_copy["input"]["context"] = item["context"]
            new_item_copy["input"]["question"] = item["question"]
            new_item_copy["input"]["rubrics"] = item["rubrics"]
            new_item_copy["input"]["answer"] = answer
            all_data.append(new_item_copy)

    with open("complete_generated_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.environ["API_KEY"]
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_globs", type=str, nargs="+", help="Glob patterns for input text files")
    argparser.add_argument("--host_url", type=str, help="Base URL for the language model API")
    args = argparser.parse_args()
    main(args, API_KEY)
