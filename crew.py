import datetime
from pathlib import Path

from crewai import Crew, Process

from agents.query_expander import create_query_expander
from agents.source_hunter import create_source_hunter
from agents.synthesizer import create_synthesizer
from agents.report_writer import create_report_writer
from tasks.task_definitions import (
    create_query_expansion_task,
    create_source_hunting_task,
    create_synthesis_task,
    create_report_writing_task,
)


def build_crew(research_question: str) -> Crew:
    """Instantiate agents and tasks, then assemble the sequential crew."""

    # --- Agents ---
    query_expander = create_query_expander()
    source_hunter = create_source_hunter()
    synthesizer = create_synthesizer()
    report_writer = create_report_writer()

    # --- Tasks (order matters: each feeds context to the next) ---
    query_expansion_task = create_query_expansion_task(
        agent=query_expander,
        research_question=research_question,
    )
    source_hunting_task = create_source_hunting_task(
        agent=source_hunter,
        query_expansion_task=query_expansion_task,
    )
    synthesis_task = create_synthesis_task(
        agent=synthesizer,
        source_hunting_task=source_hunting_task,
    )
    report_writing_task = create_report_writing_task(
        agent=report_writer,
        synthesis_task=synthesis_task,
    )

    # --- Crew ---
    crew = Crew(
        agents=[query_expander, source_hunter, synthesizer, report_writer],
        tasks=[
            query_expansion_task,
            source_hunting_task,
            synthesis_task,
            report_writing_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def run_crew(research_question: str) -> str:
    """Run the crew and save the output to outputs/."""
    crew = build_crew(research_question)
    result = crew.kickoff()

    # Save output
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # Save question for use in filename
    safe_question = (
        research_question.lower()
        .replace(" ", "_")
        .replace("?", "")
        .replace("/", "-")[:60]
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = outputs_dir / f"report_{safe_question}_{timestamp}.md"

    report_text = result if isinstance(result, str) else str(result)
    filename.write_text(report_text, encoding="utf-8")
    print(f"\n Report saved to: {filename}")

    return report_text


run_research = run_crew