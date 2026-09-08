import random
from django.shortcuts import render

def home(request):
    if request.method == "GET":
        request.session.flush()

    questions = []
    score = None

    if request.method == "POST":
        if "paragraph" in request.POST:
            request.session.flush()
            paragraph = request.POST.get("paragraph", "")
            sentences = paragraph.split(".")
            clean_sentences = [s.strip() for s in sentences if " is " in s]

            for s in clean_sentences[:5]:
                parts = s.split(" is ")
                subject = parts[0].strip().capitalize()
                definition = parts[1].strip()
                question_text = f"What is {subject}?"
                correct_answer = definition

                dummy_pool = [
                    "a database", "an operating system", "a web browser",
                    "a hardware device", "a network protocol", "a software tool"
                ]
                options = [correct_answer]
                options += random.sample(dummy_pool, 3)
                random.shuffle(options)
                correct_index = options.index(correct_answer)

                questions.append({
                    "question": question_text,
                    "options": options,
                    "correct_index": correct_index
                })

            request.session["questions"] = questions
        else:
            questions = request.session.get("questions", [])
            score = 0
            for i, q in enumerate(questions, 1):
                user_answer = request.POST.get(f"q{i}")
                if user_answer is not None and int(user_answer) == q["correct_index"]:
                    score += 1

    return render(request, "home.html", {
        "questions": request.session.get("questions", []),
        "score": score
    })
