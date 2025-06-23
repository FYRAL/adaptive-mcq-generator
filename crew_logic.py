# crew_logic.py
import os, json, re
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3
)

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        return "".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        return ""

def generate_skills_and_questions(pdf_file, num_questions):
    content = extract_text_from_pdf(pdf_file)

    # === Skill Extraction ===
    skill_agent = Agent(
        role="Skill Extractor",
        goal="Extract 5–10 concise, measurable tech skills per module from the content.",
        backstory="You are a curriculum expert.",
        verbose=False,
        allow_delegation=False,
        llm=llm,
    )

    skill_task = Task(
        description=(
            "Split the content into modules and extract 5 to 10 concise, unique, technical skills per module.\n"
            "Respond strictly in JSON format like:\n"
            "{\n"
            "  \"Module 1: Intro to Programming\": [\"Skill A\", \"Skill B\", ...],\n"
            "  \"Module 2: Web Development\": [\"Skill C\", \"Skill D\", ...]\n"
            "}\n\nContent:\n{text}"
        ),
        expected_output="JSON dictionary with module-wise skill lists.",
        agent=skill_agent,
    )

    crew1 = Crew(agents=[skill_agent], tasks=[skill_task], process=Process.sequential)
    skill_output = crew1.kickoff(inputs={"text": content})
    
    try:
        match = re.search(r"\{[\s\S]*\}", str(skill_output))
        skills_data = json.loads(match.group(0))
    except:
        return None, None, [], "Skill extraction failed"

    # Flatten skill list, limit total questions to 200 max
    flat_skills = []
    for skills in skills_data.values():
        flat_skills.extend(skills)
    if num_questions * len(flat_skills) > 200:
        flat_skills = flat_skills[:200 // num_questions]
    total_skills = len(flat_skills)
    total_mcqs = total_skills * num_questions

    # === MCQ Generation ===
    quiz_agent = Agent(
        role="Assessment Generator",
        goal="Generate multiple-choice questions from content and skills.",
        backstory="You create expert-level quizzes.",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    quiz_task = Task(
        description=(
            f"Based on this content and these skills, generate {total_mcqs} MCQs in JSON format:\n"
            "{{\"mc_questions\": [{{\"question\": \"...\", \"answers\": [\"correct\", \"wrong1\", \"wrong2\", \"wrong3\"], \"topic\": \"...\", \"difficulty\": \"easy|medium|hard\"}}]}}\n\n"
            "Content:\n{text}\n\nSkills:\n{skills}"
        ),
        expected_output="JSON with key 'mc_questions'",
        agent=quiz_agent,
    )

    crew2 = Crew(agents=[quiz_agent], tasks=[quiz_task], process=Process.sequential)
    quiz_output = crew2.kickoff(inputs={"text": content, "skills": ", ".join(flat_skills)})

    try:
        match = re.search(r"\{[\s\S]*\}", str(quiz_output))
        parsed = json.loads(match.group(0))
        mc_questions = parsed.get("mc_questions", [])
    except:
        return skills_data, total_skills, [], "Question generation failed"

    return skills_data, total_skills, mc_questions, None
