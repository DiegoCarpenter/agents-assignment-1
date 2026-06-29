from crewai import Task
from crewai import Agent


def create_query_expansion_task(agent: Agent, research_question: str) -> Task:
    return Task(
        description=(
            f"You have received the following research question:\n\n"
            f'"{research_question}"\n\n'
            "Your job is to expand this into a rich search plan. Specifically:\n\n"
            "1. Restate the core question in your own words to confirm understanding.\n"
            "2. Identify 4–6 distinct sub-questions that together cover the full scope "
            "of the research question. Each sub-question should be independently searchable.\n"
            "3. For each sub-question, list 3–5 technical keywords or phrases that are "
            "likely to appear in academic papers addressing that sub-question. Include "
            "synonyms and variant terminology where relevant (e.g., 'LLM agent', "
            "'language model agent', 'autonomous LLM').\n"
            "4. Identify any important concepts, authors, or paper titles you expect to "
            "be relevant, based on your knowledge of the AI agents literature.\n"
            "5. Note any ambiguities in the original question that the downstream agents "
            "should be aware of.\n\n"
            "Format your output as a structured document with clear headings for each "
            "section. This output will be passed directly to the Source Hunter agent, "
            "so make it easy to parse and act on."
        ),
        expected_output=(
            "A structured search plan containing:\n"
            "- Restated research question\n"
            "- 4–6 numbered sub-questions\n"
            "- For each sub-question: a keyword cluster of 3–5 terms\n"
            "- A list of anticipated relevant papers or authors\n"
            "- Any noted ambiguities or scope decisions"
        ),
        agent=agent,
    )


def create_source_hunting_task(agent: Agent, query_expansion_task: Task) -> Task:
    return Task(
        description=(
            "Using the search plan produced by the Query Expander, conduct a systematic "
            "search of the paper corpus. Follow these steps:\n\n"
            "1. For each sub-question in the plan, run at least one focused search using "
            "the search_papers tool. Use the keyword clusters as your queries — try "
            "multiple phrasings if the first query returns thin results.\n"
            "2. Also run searches for any specific papers or authors named in the plan.\n"
            "3. For each retrieved passage, record:\n"
            "   - The exact paper title and authors\n"
            "   - The publication year\n"
            "   - The key claim or finding in the passage (in your own words)\n"
            "   - A short verbatim excerpt (1–2 sentences) that supports the claim\n"
            "4. Organize your findings by sub-question.\n"
            "5. Flag any sub-questions for which you found fewer than 2 relevant passages "
            "— this signals a potential gap in the corpus.\n\n"
            "Do NOT fabricate or recall information from training. Every finding must come "
            "from a tool call result. Aim for at least 10–15 distinct passages across "
            "at least 6 different papers."
        ),
        expected_output=(
            "A structured evidence document organized by sub-question, containing:\n"
            "- For each sub-question: 2–4 relevant passages with full attribution "
            "(title, authors, year, key claim, verbatim excerpt)\n"
            "- A summary of which papers were most frequently retrieved\n"
            "- A list of any sub-questions with insufficient coverage (corpus gaps)"
        ),
        agent=agent,
        context=[query_expansion_task],
    )


def create_synthesis_task(agent: Agent, source_hunting_task: Task) -> Task:
    return Task(
        description=(
            "Using the attributed evidence collected by the Source Hunter, produce a "
            "thematic synthesis of the literature. Your work should:\n\n"
            "1. Identify 3–5 major themes that emerge across the retrieved papers. "
            "A theme is a coherent idea or debate that spans multiple papers — not "
            "just a topic label.\n"
            "2. For each theme:\n"
            "   a. Describe the central argument or finding associated with this theme.\n"
            "   b. Name the papers that contribute to it and characterize their positions.\n"
            "   c. Note where authors agree, where they diverge, and why (methodology, "
            "assumptions, scope, etc.).\n"
            "3. Identify at least 2 open questions or gaps in the literature — areas where "
            "the corpus papers disagree without resolution, or where coverage is thin.\n"
            "4. Produce a thematic outline for the literature review (section headings + "
            "1-sentence description of each section's argument).\n\n"
            "Your synthesis should read as critical analysis, not as a list of summaries. "
            "Attribute every claim to specific papers using (Author, Year) notation."
        ),
        expected_output=(
            "A synthesis document containing:\n"
            "- 3–5 named themes, each with: central argument, contributing papers and "
            "their positions, and points of agreement/disagreement\n"
            "- 2+ identified gaps or open questions with explanation\n"
            "- A thematic outline (section titles + one-line argument per section) "
            "ready for the Report Writer to expand"
        ),
        agent=agent,
        context=[source_hunting_task],
    )


def create_report_writing_task(agent: Agent, synthesis_task: Task) -> Task:
    return Task(
        description=(
            "Using the thematic synthesis, write a complete, publication-quality "
            "literature review in Markdown. The review must include all of the following "
            "sections:\n\n"
            "## Abstract\n"
            "100–150 words summarizing the question, scope, method, and key findings.\n\n"
            "## 1. Introduction\n"
            "Motivate the research question. Why does it matter? What is the scope of "
            "this review? End with a roadmap sentence.\n\n"
            "## 2. Methodology\n"
            "Briefly describe the search process: corpus used, how papers were retrieved "
            "(RAG-based search), and how themes were identified.\n\n"
            "## 3. Findings\n"
            "One subsection per theme from the synthesis (e.g., '### 3.1 Theme Name'). "
            "Each subsection should:\n"
            "- Open with a clear topic sentence stating the theme's central argument\n"
            "- Discuss contributing papers with inline citations (Author, Year)\n"
            "- Note agreements and disagreements among authors\n"
            "- Be written as flowing prose, not bullet points\n\n"
            "## 4. Discussion\n"
            "Synthesize across themes. What is the overall state of the field? "
            "Discuss the identified gaps and what future research is needed.\n\n"
            "## 5. Conclusion\n"
            "3–5 sentences summarizing the most important takeaways.\n\n"
            "## References\n"
            "List every cited paper in a consistent format:\n"
            "Author(s) (Year). *Title*. Venue.\n\n"
            "Quality standards:\n"
            "- Every factual claim must have an inline citation\n"
            "- No fabricated sources — only papers that appear in the retrieved evidence\n"
            "- Minimum 800 words in the Findings section\n"
            "- Write in clear, formal academic prose"
        ),
        expected_output=(
            "A complete Markdown literature review with all six sections: Abstract, "
            "Introduction, Methodology, Findings (with themed subsections), Discussion, "
            "Conclusion, and References. Minimum ~1200 words total. All claims cited "
            "with (Author, Year) inline. References section lists every cited paper."
        ),
        agent=agent,
        context=[synthesis_task],
    )