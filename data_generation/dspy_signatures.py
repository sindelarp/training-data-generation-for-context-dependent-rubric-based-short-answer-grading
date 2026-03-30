
import dspy

class TextCleaner(dspy.Signature):
    """
    Clean the given text by removing page numbers, repeated words, joining broken words, and removing non-semantic newlines (i.e. keep those clearly distinguishing titles or paragraphs ).
    If a short part of the text is clearly out of place, remove it.
    Remove any references to images, tables, figures, chapters, sections etc.
    Do not change the text in any other way!
    """

    text: str = dspy.InputField(description="The text to be cleaned.")
    cleaned_text: str = dspy.OutputField(description="The cleaned text.")

class QuestionGenerator(dspy.Signature):
    """
    Generate a list of 1-5 open-ended questions (more is better) asking about information contained in the given context.
    All the questions must follow these criteria:
    - It is answerable using only the information in the context.
    - It is not a subset, not standalone follow-up, or rephrasing of any of the preceding questions.
    - It is testing logical understanding of the text, not just factual recall.
    - It does not ask for anything subjective, there must always exist a single or only a few correct answers to it.
    """

    context: str = dspy.InputField(
        description="The context to generate questions from."
    )
    questions: list[str] = dspy.OutputField(
        description="List of the generated questions.",
    )


class QuestionCritic(dspy.Signature):
    """
    Given a list of questions and a context, for each question decide whether it follows these criteria:
    - It is answerable using only the information in the context.
    - It is not a subset, non standalone follow-up, or rephrasing of any of the preceding questions.
    - It is testing logical understanding of the text, not just factual recall.
    - It does not ask for anything subjective, there must always exist a single or only a few correct answers to it.
    """

    context: str = dspy.InputField(description="The context for the questions.")
    questions: list[str] = dspy.InputField(description="List of the questions.")
    critiques: list[bool] = dspy.OutputField(
        description="List of the critiques for each question, True if the question follows all the criteria, False if it fails at least one of the criteria.",
    )


class GenerateGradingRubric(dspy.Signature):
    """
    Generate a 3 point grading rubric for the given question and context that follows these criteria:
    - The full credit section describes a fully correct answer to the question.
    - The partial credit section describes an answer that is not fully correct (unambiguously), but shows at least partially correct reasoning and understanding of the context and the question.
    - The no credit section describes an answer that is fully incorrect or shows little to no correct reasoning and understanding of the context and the question.
    - The individual sections do not have any overlap creating ambiguity in grading.
    - The sections are similar in detail and quality to the following examples:
        Question:
            A large consumer group focused on promoting ethical clothing production has called for a ban on Fast Fashion clothes. They are urging people to buy durable clothing instead. Their campaign has been very successful on social media and has millions of followers committed to the ban. There can be both positive and negative consequences when people buy more durable clothes and less Fast Fashion.\n\nDescribe one positive consequence and one negative consequence.
        Full Credit:
            Includes a correct response for both the positive AND negative consequences. In general, the two consequences should appear in the correct boxes. If the student puts both responses in a single box, it must be clear that one is a positive consequence and one is a negative consequence. Correct possible consequences for each are provided below.\n\nPositive consequences – Responses should refer to one of the following types of positive consequences:\n\n1. Positive effects on the environment\n2. Positive effects on workers\n3. Positive effects for customers\n4. Positive effects for the clothing production industry\n5. Positive effects on fashion or clothing\n\nSample responses:\n\n- People would wear durable clothes more often and less clothes would be thrown away. [1]\n- There would be less pollution. [1]\n- Ethical clothing reduces waste in landfills. [1]\n- Workers get fair wages. [2]\n- Workers will be treated better. [2]\n- There would be more durable clothing available. [3 or 5]\n- More durable clothing options might increase competition. [3 or 4]\n- More durable clothing factories will open. [4]\n- It could become easier and cheaper to make durable clothing. [3 or 4]\n- It will be more ethical. [5]\n- Clothing will last longer. [5]\n\nNegative consequences – Responses should refer to one of the following types of consequences:\n\n1. Negative effects on workers\n2. Negative effects on customers or clothing charities\n3. Negative effects on the clothing production industry\n4. Negative effects on fashion or clothing\n\nSample responses:\n\n- Some Fast Fashion factories might close because people don’t buy the clothes. [1 or 3]\n- People won’t have to buy as many clothes because durable clothes last longer, so there will be fewer jobs for clothing workers. [1]\n- Clothing prices could go up for everyone if there is more durable clothing available than Fast Fashion. [2]\n- There will be fewer style options. [2 or 4]\n- Fewer clothes will be donated to charity. [2]\n- Companies will make less profit with durable clothes. [3]\n- If companies switch to durable clothes, they might not be as successful as they were before. [3]\n- Clothes will be more boring.
        Partial Credit:
            Includes a correct possible positive consequence OR a correct possible negative consequence. The other possible consequence is missing, incorrect, vague, insufficient or irrelevant. The consequences are vaguely described.
        No Credit:
            The answer misunderstands the point of the article or the question or does not make sense when only context is the article.

        Question:
            In the last paragraph of the blog, the professor writes: \\Another mystery remained...\\\"\nTo what mystery does she refer?\"
        Full Credit:
            Responses refer to the disappearance of the materials used to move the statues (moai). Eg.: \n What happened to these plants and large trees that had been used to move the moai? [Direct\nquotation]\n There are no large trees left that could have moved the moai.\n There are grasses, shrubs and some small trees, but no trees large enough to move the\nlarge statues.\n Where are the large trees? What happened to the resources that were needed to transport the statues?\n She was referring to what moved the Moai because when she looked around there were no\nbig trees or plants. She is also wondering what happened to them. [Although this response\nbegins by referring to the wrong mystery, it contains the correct elements.]
        Partial Credit:
            The answer does not specify the exact kind of vegetation missing. Eg.: Where are the plants?
        No Credit:
            The answer misunderstands the point of the article or the question or does not make sense when only context is the article.

        Question:
            Why are the characters acting so strangely when introducing themselves?
        Full Credit:
            References either to the fact that they are pretending they are in a play or to the fact that the author is introducing these characters.
        Partial Credit:
            Refers to some vaguer reason, ie. because they just drank wine.
        No Credit:
            Refers to some fact not mentioned in the text or a part that is irrelevant.

    You may optionally also receive examples showing grading rubric sections that do not follow some of these criteria and written critique of what is wrong with what rubric section. If received, use these as the base for your response.
    """

    context: str = dspy.InputField(description="The context for the question.")
    question: str = dspy.InputField(
        description="The question to generate a grading rubric for."
    )
    full_credit_example: str | None = dspy.InputField(
        description="Example description of what a full credit answer should look like."
    )
    partial_credit_example: str | None = dspy.InputField(
        description="Example description of what a partial credit answer should look like."
    )
    no_credit_example: str | None = dspy.InputField(
        description="Example description of what a no credit answer should look like."
    )
    example_critique: str | None = dspy.InputField(
        description="Critique of the given example rubric sections."
    )
    full_credit: str = dspy.OutputField(
        description="Description of what a full credit answer should look like."
    )
    partial_credit: str = dspy.OutputField(
        description="Description of what a partial credit answer should look like."
    )
    no_credit: str = dspy.OutputField(
        description="Description of what a no credit answer should look like."
    )


class GradingRubricCritic(dspy.Signature):
    """
    Given a context and a question regarding it, decide whether the 3 point grading rubric follows these criteria:
    - The full credit section describes a fully correct answer to the question.
    - The partial credit section describes an answer that is not fully correct (unambiguously), but shows at least partially correct reasoning and understanding of the context and the question.
    - The no credit section describes an answer that is fully incorrect or shows little to no correct reasoning and understanding of the context and the question.
    - The individual sections do not have any overlap creating ambiguity in grading.
    - The sections are similar in detail and quality to the following examples:
        Question:
            A large consumer group focused on promoting ethical clothing production has called for a ban on Fast Fashion clothes. They are urging people to buy durable clothing instead. Their campaign has been very successful on social media and has millions of followers committed to the ban. There can be both positive and negative consequences when people buy more durable clothes and less Fast Fashion.\n\nDescribe one positive consequence and one negative consequence.
        Full Credit:
            Includes a correct response for both the positive AND negative consequences. In general, the two consequences should appear in the correct boxes. If the student puts both responses in a single box, it must be clear that one is a positive consequence and one is a negative consequence. Correct possible consequences for each are provided below.\n\nPositive consequences – Responses should refer to one of the following types of positive consequences:\n\n1. Positive effects on the environment\n2. Positive effects on workers\n3. Positive effects for customers\n4. Positive effects for the clothing production industry\n5. Positive effects on fashion or clothing\n\nSample responses:\n\n- People would wear durable clothes more often and less clothes would be thrown away. [1]\n- There would be less pollution. [1]\n- Ethical clothing reduces waste in landfills. [1]\n- Workers get fair wages. [2]\n- Workers will be treated better. [2]\n- There would be more durable clothing available. [3 or 5]\n- More durable clothing options might increase competition. [3 or 4]\n- More durable clothing factories will open. [4]\n- It could become easier and cheaper to make durable clothing. [3 or 4]\n- It will be more ethical. [5]\n- Clothing will last longer. [5]\n\nNegative consequences – Responses should refer to one of the following types of consequences:\n\n1. Negative effects on workers\n2. Negative effects on customers or clothing charities\n3. Negative effects on the clothing production industry\n4. Negative effects on fashion or clothing\n\nSample responses:\n\n- Some Fast Fashion factories might close because people don’t buy the clothes. [1 or 3]\n- People won’t have to buy as many clothes because durable clothes last longer, so there will be fewer jobs for clothing workers. [1]\n- Clothing prices could go up for everyone if there is more durable clothing available than Fast Fashion. [2]\n- There will be fewer style options. [2 or 4]\n- Fewer clothes will be donated to charity. [2]\n- Companies will make less profit with durable clothes. [3]\n- If companies switch to durable clothes, they might not be as successful as they were before. [3]\n- Clothes will be more boring.
        Partial Credit:
            Includes a correct possible positive consequence OR a correct possible negative consequence. The other possible consequence is missing, incorrect, vague, insufficient or irrelevant. The consequences are vaguely described.
        No Credit:
            The answer misunderstands the point of the article or the question or does not make sense when only context is the article.

        Question:
            In the last paragraph of the blog, the professor writes: \\Another mystery remained...\\\"\nTo what mystery does she refer?\"
        Full Credit:
            Responses refer to the disappearance of the materials used to move the statues (moai). Eg.: \n What happened to these plants and large trees that had been used to move the moai? [Direct\nquotation]\n There are no large trees left that could have moved the moai.\n There are grasses, shrubs and some small trees, but no trees large enough to move the\nlarge statues.\n Where are the large trees? What happened to the resources that were needed to transport the statues?\n She was referring to what moved the Moai because when she looked around there were no\nbig trees or plants. She is also wondering what happened to them. [Although this response\nbegins by referring to the wrong mystery, it contains the correct elements.]
        Partial Credit:
            The answer does not specify the exact kind of vegetation missing. Eg.: Where are the plants?
        No Credit:
            The answer misunderstands the point of the article or the question or does not make sense when only context is the article.

        Question:
            Why are the characters acting so strangely when introducing themselves?
        Full Credit:
            References either to the fact that they are pretending they are in a play or to the fact that the author is introducing these characters.
        Partial Credit:
            Refers to some vaguer reason, ie. because they just drank wine.
        No Credit:
            Refers to some fact not mentioned in the text or a part that is irrelevant.
    """

    context: str = dspy.InputField(description="The context for the question.")
    question: str = dspy.InputField(
        description="The question to generate a grading rubric for."
    )
    full_credit: str = dspy.InputField(
        description="Description of what a full credit answer should look like."
    )
    partial_credit: str = dspy.InputField(
        description="Description of what a partial credit answer should look like."
    )
    no_credit: str = dspy.InputField(
        description="Description of what a no credit answer should look like."
    )
    follows_criteria: bool = dspy.OutputField(
        description="True if all of the sections follow the criteria, False otherwise."
    )
    critique: str | None = dspy.OutputField(
        description="If the question follows the criteria this field should be None, otherwise it should explain which section(s) don't meet what criteria and why."
    )


class GenerateCorrectAnswer(dspy.Signature):
    """
    Generate a list of 5 correct answers for the given context, question regarding and a grading rubric section.
    All the answers must follow these criteria:
    - It is a fully correct answer to the question based on the information from the context, meaning it follows the grading criteria for correct answers.
    - It is not a copy of a any of the previous answers, it can contain the same information, but it must be worded differently or include/exclude additional information that does not change the correctness of the answer.
    - It is written as a high schooler might write it under time pressure.
    - It does not appear AI generated, it may contain common typos like their/they're errors or slightly awkward phrasing.
    """

    context: str = dspy.InputField(description="The context for the question.")
    question: str = dspy.InputField(
        description="The question to generate an answer for."
    )
    grading_criteria: str = dspy.InputField(
        description="The grading criteria the answers should satisfy."
    )
    answers: list[str] = dspy.OutputField(
        description="A list of 5 answers following the criteria."
    )


class CorrectAnswerCritic(dspy.Signature):
    """
    Given a list of answers, context, question regarding the context and grading rubric section, for each answer decide whether it follows these criteria:
    - It is a fully correct answer to the question based on the information from the context, meaning it follows the grading criteria for correct answers.
    - It is not a copy of a any of the previous answers, it can contain the same information, but it must be worded differently or include/exclude additional information that does not change the correctness of the answer. If a previous answer was rejected because it does not meet some of the criteria, one of the next answers can just be a correction.
    - It is written as a high schooler might write it under time pressure.
    - It does not appear AI generated, it may contain common typos like their/they're errors or slightly awkward phrasing.
    """

    context: str = dspy.InputField(description="The context for the question.")
    question: str = dspy.InputField(
        description="The question to generate an answer for."
    )
    grading_criteria: str = dspy.InputField(
        description="The grading criteria the answers should satisfy."
    )
    answers: list[str] = dspy.InputField(description="List of the answers..")
    critiques: list[bool] = dspy.OutputField(
        description="List of the critiques for each answer, True if the answer follows all the criteria, False if it fails at least one of the criteria.",
    )


class GenerateSemicorrectAnswer(dspy.Signature):
    """
    Generate a list of 5 partially correct answers for the given context, question regarding and a grading rubric section.
    All the answers must follow these criteria:
    - It is a partially correct answer to the question based on the information from the context, meaning it follows the grading criteria for partially correct answers.
    - The incorrectness comes from a possible misinterpretation of the text or simply excluding information required to be fully correct.
    - It is not a copy of a any of the previous answers, it can contain the same information, but it must be worded differently or include/exclude additional information that does not change the correctness of the answer.
    - It is written as a high schooler might write it under time pressure.
    - It does not appear AI generated, it may contain common typos like their/they're errors or slightly awkward phrasing.
    """

    context: str = dspy.InputField(description="The context for the question.")
    question: str = dspy.InputField(
        description="The question to generate an answer for."
    )
    grading_criteria: str = dspy.InputField(
        description="The grading criteria the answers should satisfy."
    )
    answers: list[str] = dspy.OutputField(
        description="A list of 5 answers following the criteria."
    )


class SemicorrectAnswerCritic(dspy.Signature):
    """
    Given a list of answers, context, question regarding the context and grading rubric section, for each answer decide whether it follows these criteria:
    - It is a partially correct answer to the question based on the information from the context, meaning it follows the grading criteria for partially correct answers.
    - The incorrectness comes from a possible misinterpretation of the text or simply excluding information required to be fully correct.
    - It is not a copy of a any of the previous answers, it can contain the same information, but it must be worded differently or include/exclude additional information that does not change the correctness of the answer. If a previous answer was rejected because it does not meet some of the criteria, one of the next answers can just be a correction.
    - It is written as a high schooler might write it under time pressure.
    - It does not appear AI generated, it may contain common typos like their/they're errors or slightly awkward phrasing.
    """

    context: str = dspy.InputField(description="The context for the question.")
    question: str = dspy.InputField(
        description="The question to generate an answer for."
    )
    grading_criteria: str = dspy.InputField(
        description="The grading criteria the answers should satisfy."
    )
    answers: list[str] = dspy.InputField(description="List of the answers..")
    critiques: list[bool] = dspy.OutputField(
        description="List of the critiques for each answer, True if the answer follows all the criteria, False if it fails at least one of the criteria.",
    )


class GenerateIncorrectAnswer(dspy.Signature):
    """
    Generate a list of 5 incorrect answers for the given context, question regarding and a grading rubric section.
    All the answers must follow these criteria:
    - It is an incorrect answer to the question based on the information from the context, meaning it follows the grading criteria for incorrect answers.
    - The incorrectness comes from a possible misinterpretation of the text, simply excluding correct information or trying to "talk" around the question without really answering it.
    - It is not a copy of a any of the previous answers, it can contain the same information, but it must be worded differently or include/exclude additional information that does not change the correctness of the answer.
    - It is written as a high schooler might write it under time pressure.
    - It does not appear AI generated, it may contain common typos like their/they're errors or slightly awkward phrasing.
    """

    context: str = dspy.InputField(description="The context for the question.")
    question: str = dspy.InputField(
        description="The question to generate an answer for."
    )
    grading_criteria: str = dspy.InputField(
        description="The grading criteria the answers should satisfy."
    )
    answers: list[str] = dspy.OutputField(
        description="A list of 5 answers following the criteria."
    )


class IncorrectAnswerCritic(dspy.Signature):
    """
    Given a list of answers, context, question regarding the context and grading rubric section, for each answer decide whether it follows these criteria:
    - It is an incorrect answer to the question based on the information from the context, meaning it follows the grading criteria for incorrect answers.
    - The incorrectness comes from a possible misinterpretation of the text, simply excluding correct information or trying to "talk" around the question without really answering it.
    - It is not a copy of a any of the previous answers, it can contain the same information, but it must be worded differently or include/exclude additional information that does not change the correctness of the answer. If a previous answer was rejected because it does not meet some of the criteria, one of the next answers can just be a correction.
    - It is written as a high schooler might write it under time pressure.
    - It does not appear AI generated, it may contain common typos like their/they're errors or slightly awkward phrasing.
    """

    context: str = dspy.InputField(description="The context for the question.")
    question: str = dspy.InputField(
        description="The question to generate an answer for."
    )
    grading_criteria: str = dspy.InputField(
        description="The grading criteria the answers should satisfy."
    )
    answers: list[str] = dspy.InputField(description="List of the answers..")
    critiques: list[bool] = dspy.OutputField(
        description="List of the critiques for each answer, True if the answer follows all the criteria, False if it fails at least one of the criteria.",
    )
