import json
import os
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# --- CONFIG ---
MY_GROQ_KEY = os.getenv("GROQ_API_KEY")
TRANSCRIPT_FOLDER = "./nptel_transcripts/"
OUTPUT_FILE = "study_guide.json"

summary_template = """
You are an expert NPTEL professor creating an official study guide for the 'Introduction to IoT' course.
Your task is to create a high-quality, **detailed, and comprehensive** summary of the *entire* lecture transcript provided.

The student reading this is cramming for their final exam and may not have time to watch the video. Your summary must be good enough to be their primary study material for this lecture.

**Instructions:**
1. Start with a 2-3 sentence overview of the lecture's main topic.
2. Create clear sections for each major sub-topic discussed. Use **Markdown headings** (e.g., `## Key Definitions`, `## CoAP Protocol Explained`).
3. Under each heading, use **detailed bullet points** to extract:
   * All key definitions (e.g., "IoT:", "M2M:").
   * Important concepts, technical terms, and protocols (e.g., "REST Architecture", "MQTT").
   * Any comparisons made (e.g., "HTTP vs. CoAP").
   * Lists of features, characteristics, or enablers.
4. The summary must be structured, easy to read, and cover all essential information from the transcript. Do not be "concise"; be "comprehensive".

**Transcript:**
{context}

**Comprehensive Exam Study Guide:**
"""
summary_prompt = ChatPromptTemplate.from_template(summary_template)


def get_sort_key(filename: str) -> tuple[int, int]:
    """Extract (week, lecture) numbers from a filename like 'W10_L46.txt'."""
    numbers = re.findall(r"\d+", filename)
    if len(numbers) >= 2:
        return (int(numbers[0]), int(numbers[1]))
    if len(numbers) == 1:
        return (int(numbers[0]), 0)
    return (999, 999)


def main() -> None:
    if not MY_GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment or .env file.")

    llm = ChatGroq(model="openai/gpt-oss-20b", api_key=MY_GROQ_KEY)
    summary_chain = summary_prompt | llm | StrOutputParser()

    all_files = [f for f in os.listdir(TRANSCRIPT_FOLDER) if f.endswith(".txt")]
    all_files.sort(key=get_sort_key)
    print(f"Found {len(all_files)} files. Generating study guide...")

    study_guide = {}
    for filename in all_files:
        file_path = os.path.join(TRANSCRIPT_FOLDER, filename)
        source_name = filename.replace(".txt", "").replace("_", " ")
        print(f"Processing: {source_name}...")

        loader = TextLoader(file_path, encoding="utf-8")
        doc = loader.load()[0]

        try:
            summary = summary_chain.invoke({"context": doc.page_content})
            study_guide[source_name] = summary
            print("-> Done.")
        except Exception as e:
            print(f"-> ERROR processing {filename}: {e}")
            print("   Skipping this file.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(study_guide, f, indent=4, ensure_ascii=False)

    print(f"\nSuccess! Study guide saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
