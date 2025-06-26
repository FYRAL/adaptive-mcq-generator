import os
import json
import re
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

app = FastAPI()

# PDF Extraction 
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3
)


MAX_MCQS = 200
CHUNK_SIZE = 10  # skills per chunk

@app.post("/run-crew/")
async def run_crew(file: UploadFile = File(...), num_questions: int = 2):
    try:
        content = extract_text_from_pdf(file.file)
        if not content:
            raise HTTPException(status_code=400, detail="Failed to extract content from PDF.")

        #  Step 1: Extract skills per module 
        skill_agent = Agent(
            role="Skill Extractor",
            goal="Extract key skills per module from the learning content",
            backstory="Expert instructional designer",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        skill_task = Task(
            description=(
                "From the following course content, extract skills grouped by modules."
                " Each module should have 5 to 10 concise, technical skills."
                " Respond in JSON like: {\"Module X: Title\": [\"Skill 1\", ...]}\n\nContent:\n{text}"
            ),
            expected_output="A JSON with modules and skill lists.",
            agent=skill_agent,
        )

        crew1 = Crew(
            agents=[skill_agent],
            tasks=[skill_task],
            process=Process.sequential,
            verbose=True,
        )

        skill_output = crew1.kickoff(inputs={"text": content})
        skill_json_str = re.search(r"```json\s*(.*?)```", str(skill_output), re.DOTALL)
        if skill_json_str:
            skills_data = json.loads(skill_json_str.group(1).strip())
        else:
            skills_data = json.loads(str(skill_output))

        # Step 2: Filter and limit skills 
        filtered_skills = {
            module: skills[:10] for module, skills in skills_data.items()
            if 5 <= len(skills) <= 10
        }

        flat_skills = [
            {"module": module, "skill": skill}
            for module, skills in filtered_skills.items()
            for skill in skills
        ]

        max_skills = min(len(flat_skills), MAX_MCQS // num_questions)
        flat_skills = flat_skills[:max_skills]

        total_skills = len(flat_skills)
        total_mcqs_expected = total_skills * num_questions

        # Step 3: Chunk skills 
        chunks = [
            flat_skills[i:i+CHUNK_SIZE] for i in range(0, len(flat_skills), CHUNK_SIZE)
        ]

        # Step 4: Generate MCQs 
        quiz_agent = Agent(
            role="Assessment Generator",
            goal="Generate MCQs from skills and content",
            backstory="Experienced learning assessment expert",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        all_questions = []
        for chunk in chunks:
            skill_lines = "\n".join([f"- {s['skill']} ({s['module']})" for s in chunk])
            quiz_task = Task(
                description=(
                    f"Generate {num_questions * len(chunk)} MCQs based on the following skills."
                    f" Each skill should inspire {num_questions} questions."
                    f" Respond in JSON: {{\"mc_questions\": [{{\"question\": \"...\", \"answers\": [\"correct\", \"wrong1\", ...], \"topic\": \"...\", \"difficulty\": \"...\"}}, ...]}}\n\n"
                    f"Skills:\n{skill_lines}\n\nContent:\n{content[:1000]}"
                ),
                expected_output="A JSON with a 'mc_questions' key holding the questions.",
                agent=quiz_agent,
            )

            crew2 = Crew(
                agents=[quiz_agent],
                tasks=[quiz_task],
                process=Process.sequential,
                verbose=True
            )

            quiz_result = crew2.kickoff()
            match = re.search(r"```json\s*(.*?)```", str(quiz_result), re.DOTALL)
            try:
                quiz_json = json.loads(match.group(1) if match else str(quiz_result))
                all_questions.extend(quiz_json.get("mc_questions", []))
            except Exception as e:
                print("JSON parsing error in quiz_result chunk:\n", str(quiz_result)[:1000])

        return JSONResponse(content={
            "skills": filtered_skills,
            "total_skills": total_skills,
            "total_mcqs_expected": total_mcqs_expected,
            "actual_mcqs": len(all_questions),
            "mc_questions": all_questions
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
