from langchain_core.prompts import (
    ChatPromptTemplate,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)

from tools.prompts.report_refinement_tools_prompts import (
    REPORT_REFINEMENT_CLASSIFY_INTENT_SYSTEM_PROMPT,
    REPORT_REFINEMENT_NEW_SECTION_SYSTEM_PROMPT,
    REPORT_REFINEMENT_PLACEMENT_SYSTEM_PROMPT,
    REPORT_REFINEMENT_SECTION_EDIT_SYSTEM_PROMPT,
)
from tools.search_tools import SearchTools
from tools.report_section_tools import (
    ReportSectionTools,
)

from state.models import SubQuery
from state.constants import Domain

from utils.vector_store import VectorStoreManager
from langchain_core.documents import Document as LangchainDocument
from uuid import uuid4


class ReportRefinementTools:
    """Tools for section-aware report refinement."""

    MAX_SECTION_LENGTH = 4000

    MAX_REFINEMENT_QUERY = 1000

    MAX_SECTION_CONTEXT = 2500

    @staticmethod
    def classify_refinement_intent(
        refinement_query: str,
        existing_sections: list[str],
        report_title: str = "",
    ) -> dict:
        """
        Detect refinement intent safely.
        """

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    (
                        "system",
                        REPORT_REFINEMENT_CLASSIFY_INTENT_SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        (
                            "Report Title:\n"
                            "{title}\n\n"

                            "Refinement Request:\n"
                            "{query}"
                        ),
                    ),
                ]
            )
        )

        chain = (
            prompt
            | llm
        )

        response = chain.invoke(
            {
                "query":
                refinement_query[
                    :ReportRefinementTools
                    .MAX_REFINEMENT_QUERY
                ],

                "sections":
                ", ".join(
                    existing_sections
                ),
                "title": report_title,
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(response.content),
                fallback={
                    "intent": "modify_section",
                    "target_section": (
                        existing_sections[0] if existing_sections else ""
                    ),
                    "search_query": "",
                },
            )
        )

        target_section = (
            parsed_response.get(
                "target_section",
                "",
            )
        )

        intent = (
            parsed_response.get(
                "intent",
                "modify_section",
            )
        )

        # Ensure search_query exists
        if "search_query" not in parsed_response:
            parsed_response["search_query"] = ""

        # Safety correction

        if (
            intent == "modify_section"
            and target_section
            not in existing_sections
        ):

            parsed_response[
                "target_section"
            ] = (
                existing_sections[0]
                if existing_sections
                else ""
            )

        return parsed_response

    @staticmethod
    def refine_section(
        section_name: str,
        section_content: str,
        refinement_query: str,
        report_title: str = "",
        session_id: str = "",
        retrieved_documents: list[dict] | None = None,
    ) -> str:
        """
        Refine ONLY one section.
        """

        section_content = (
            section_content[
                :ReportRefinementTools
                .MAX_SECTION_CONTEXT
            ]
        )

        refinement_query = (
            refinement_query[
                :ReportRefinementTools
                .MAX_REFINEMENT_QUERY
            ]
        )

        evidence = ""

        # If retrieved_documents provided by the retrieval agent, use them
        if retrieved_documents:
            try:
                entries = []
                for d in retrieved_documents[:5]:
                    title = d.get("title") or d.get("section") or ""
                    src = d.get("source", "external")
                    content = d.get("content") or (d.get("document").page_content if d.get("document") else "")
                    snippet = (content or "")[:1200]
                    entries.append(f"{title} — {src}\n{snippet}\n\n")
                evidence = "".join(entries)
            except Exception:
                evidence = ""

        # Otherwise, fall back to session_id-driven inline search (older behavior)
        elif session_id:
            try:
                query = f"{report_title} {section_name} {refinement_query} {section_content[:300]}".strip()
                sub_query = SubQuery(query=query, domain=Domain.WEB, priority=3)

                candidates: list = []
                try:
                    candidates += SearchTools.search_web(sub_query, max_results=3)
                except Exception:
                    pass

                try:
                    candidates += SearchTools.search_wikipedia(sub_query)
                except Exception:
                    pass

                docs = SearchTools.deduplicate_documents(candidates)

                try:
                    vector_store = VectorStoreManager.get_vector_store(session_id=session_id)
                    lc_docs = []
                    ids = []
                    for d in docs:
                        content = d.content if isinstance(d.content, str) else str(d.content)
                        if not content.strip():
                            continue
                        lc_docs.append(
                            LangchainDocument(
                                page_content=content,
                                metadata={
                                    "chunk_type": "external",
                                    "source": d.source.value if hasattr(d.source, "value") else str(d.source),
                                    "title": d.title,
                                    "url": d.url,
                                    "session_id": session_id,
                                },
                            )
                        )
                        ids.append(str(uuid4()))

                    if lc_docs:
                        vector_store.add_documents(documents=lc_docs, ids=ids)
                except Exception:
                    pass

                # Build evidence
                evidence = ""
                for d in docs[:3]:
                    title = d.title or ""
                    src = d.source.value if hasattr(d.source, "value") else str(d.source)
                    snippet = (d.content or "")[:1200]
                    evidence += f"{title} — {src}\n{snippet}\n\n"
            except Exception:
                evidence = ""

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    ("system",
                        REPORT_REFINEMENT_SECTION_EDIT_SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        (
                            "Section Name:\n"
                            "{section_name}\n\n"

                            "Existing Section Content:\n"
                            "{section_content}\n\n"

                            "User Refinement Request:\n"
                            "{refinement_query}"
                        ),
                    ),
                ]
            )
        )

        chain = (
            prompt
            | llm
        )

        # Inject retrieved evidence into the section content to allow RAG
        section_content_for_llm = (
            section_content
            + ("\n\nRetrieved Evidence:\n" + evidence if evidence else "")
        )

        response = chain.invoke(
            {
                "section_name": section_name,
                "section_content": section_content_for_llm,
                "refinement_query": refinement_query,
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "updated_section":
                    section_content,
                },
            )
        )

        updated_section = (
            parsed_response.get(
                "updated_section",
                section_content,
            )
        )

        if not isinstance(
            updated_section,
            str,
        ):

            updated_section = str(
                updated_section
            )

        # Sanitize model output: if the LLM returned a full report,
        # try to extract just the targeted section. Also remove
        # accidental leading H1/H2 headings or repeated inline titles.
        try:
            cleaned = updated_section.strip()

            # If the model returned a full report (contains multiple ## headings),
            # extract the section matching section_name.
            extracted = ReportSectionTools.extract_sections(cleaned)
            normalized = ReportSectionTools.normalize_section_name(section_name)
            if normalized in extracted.get("sections", {}):
                cleaned = extracted["sections"][normalized]
            else:
                # Remove any leading document-level title lines (e.g. "# Title")
                cleaned = cleaned.lstrip()
                # Normalize newlines
                cleaned = cleaned.replace("\r\n", "\n")

                # Strip any leading H1/H2 headings
                cleaned_lines = cleaned.splitlines()
                while cleaned_lines and cleaned_lines[0].strip().startswith("#"):
                    cleaned_lines = cleaned_lines[1:]

                cleaned = "\n".join(cleaned_lines).lstrip()

                # remove leading duplicated inline section title
                first_split = cleaned.split("\n", 1)
                if len(first_split) > 1:
                    first_line = first_split[0].strip().lower()
                    norm_title = section_name.strip().lower()
                    if first_line == norm_title:
                        cleaned = first_split[1].lstrip()

            updated_section = cleaned
        except Exception:
            updated_section = str(updated_section).strip()

        return (
            updated_section
            .strip()[
                :ReportRefinementTools
                .MAX_SECTION_LENGTH
            ]
        )

    @staticmethod
    def generate_new_section(
        section_name: str,
        report_topic: str,
        refinement_query: str,
        existing_sections: list[str],
        report_title: str = "",
        session_id: str = "",
        retrieved_documents: list[dict] | None = None,
    ) -> str:
        """
        Generate lightweight new section.
        """

        refinement_query = (
            refinement_query[
                :ReportRefinementTools
                .MAX_REFINEMENT_QUERY
            ]
        )

        evidence = ""

        # Use retrieved documents from retrieval agent if provided
        if retrieved_documents:
            try:
                entries = []
                for d in retrieved_documents[:6]:
                    title = d.get("title") or d.get("section") or ""
                    src = d.get("source", "external")
                    content = d.get("content") or (d.get("document").page_content if d.get("document") else "")
                    snippet = (content or "")[:1200]
                    entries.append(f"{title} — {src}\n{snippet}\n\n")
                evidence = "".join(entries)
            except Exception:
                evidence = ""

        # Otherwise fall back to inline session_id-based search
        elif session_id:
            try:
                query = f"{report_title} {section_name} {refinement_query} {report_topic[:300]}".strip()
                sub_query = SubQuery(query=query, domain=Domain.WEB, priority=3)

                candidates: list = []
                try:
                    candidates += SearchTools.search_web(sub_query, max_results=4)
                except Exception:
                    pass

                try:
                    candidates += SearchTools.search_wikipedia(sub_query)
                except Exception:
                    pass

                docs = SearchTools.deduplicate_documents(candidates)

                try:
                    vector_store = VectorStoreManager.get_vector_store(session_id=session_id)
                    lc_docs = []
                    ids = []
                    for d in docs:
                        content = d.content if isinstance(d.content, str) else str(d.content)
                        if not content.strip():
                            continue
                        lc_docs.append(
                            LangchainDocument(
                                page_content=content,
                                metadata={
                                    "chunk_type": "external",
                                    "source": d.source.value if hasattr(d.source, "value") else str(d.source),
                                    "title": d.title,
                                    "url": d.url,
                                    "session_id": session_id,
                                },
                            )
                        )
                        ids.append(str(uuid4()))

                    if lc_docs:
                        vector_store.add_documents(documents=lc_docs, ids=ids)
                except Exception:
                    pass

                evidence = ""
                for d in docs[:4]:
                    title = d.title or ""
                    src = d.source.value if hasattr(d.source, "value") else str(d.source)
                    snippet = (d.content or "")[:1200]
                    evidence += f"{title} — {src}\n{snippet}\n\n"
            except Exception:
                evidence = ""

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    ("system",
                        REPORT_REFINEMENT_NEW_SECTION_SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        (
                            "Report Topic:\n"
                            "{topic}\n\n"

                            "Existing Sections:\n"
                            "{existing_sections}\n\n"

                            "New Section Name:\n"
                            "{section_name}\n\n"

                            "User Request:\n"
                            "{refinement_query}"
                        ),
                    ),
                ]
            )
        )

        chain = (
            prompt
            | llm
        )

        # Inject retrieved evidence into the refinement query for RAG
        refinement_with_evidence = (
            refinement_query + "\n\nRetrieved Evidence:\n" + evidence
            if evidence
            else refinement_query
        )

        response = chain.invoke(
            {
                "topic": report_topic,
                "existing_sections": ", ".join(existing_sections),
                "section_name": section_name,
                "refinement_query": refinement_with_evidence,
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "new_section":
                    (
                        "This section discusses "
                        f"{section_name.replace('_', ' ')} "
                        "in the context of the "
                        "research topic."
                    ),
                },
            )
        )

        new_section = (
            parsed_response.get(
                "new_section",
                "",
            )
        )

        if not new_section:

            new_section = (
                "Additional discussion on "
                f"{section_name.replace('_', ' ')}."
            )

        if not isinstance(
            new_section,
            str,
        ):

            new_section = str(
                new_section
            )

        # Sanitize generated new section similarly to refined sections.
        try:
            cleaned = new_section.strip()
            extracted = ReportSectionTools.extract_sections(cleaned)
            normalized = ReportSectionTools.normalize_section_name(section_name)
            if normalized in extracted.get("sections", {}):
                cleaned = extracted["sections"][normalized]
            else:
                cleaned = cleaned.replace("\r\n", "\n")
                cleaned_lines = cleaned.splitlines()
                while cleaned_lines and cleaned_lines[0].strip().startswith("#"):
                    cleaned_lines = cleaned_lines[1:]
                cleaned = "\n".join(cleaned_lines).lstrip()
                first_split = cleaned.split("\n", 1)
                if len(first_split) > 1:
                    first_line = first_split[0].strip().lower()
                    norm_title = section_name.strip().lower()
                    if first_line == norm_title:
                        cleaned = first_split[1].lstrip()
            new_section = cleaned
        except Exception:
            new_section = str(new_section).strip()

        return (
            new_section
            .strip()[
                :ReportRefinementTools
                .MAX_SECTION_LENGTH
            ]
        )

    @staticmethod
    def determine_section_placement(
        new_section: str,
        existing_sections: list[str],
    ) -> dict:
        """
        Determine intelligent placement.
        """

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    ("system",
                        REPORT_REFINEMENT_PLACEMENT_SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        (
                            "Existing Sections:\n"
                            "{existing_sections}\n\n"

                            "New Section:\n"
                            "{new_section}"
                        ),
                    ),
                ]
            )
        )

        chain = (
            prompt
            | llm
        )

        response = chain.invoke(
            {
                "new_section":
                new_section,

                "existing_sections":
                ", ".join(
                    existing_sections
                ),
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "insert_before":
                    existing_sections[-1]
                    if existing_sections
                    else "",
                },
            )
        )

        return parsed_response