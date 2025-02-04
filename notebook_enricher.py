import json
from typing import Dict, List, Any, Tuple, Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from tqdm import tqdm

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    SafetySetting,
)

# Constants for model and safety settings
MODEL_ID = "gemini-2.0-flash-exp"

# JSON schema for notebook structure analysis
NOTEBOOK_STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "introduction": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "start_cell": {"type": "integer"},
                    "end_cell": {"type": "integer"},
                },
                "required": ["title", "start_cell", "end_cell"],
            },
        },
    },
    "required": ["title", "sections", "introduction"],
}

GENERATION_CONFIG = GenerateContentConfig(
    max_output_tokens=8192,
    temperature=0.1,
    top_k=40,
    top_p=0.95,
    safety_settings=[
        SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF",
        ),
        SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF",
        ),
        SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF",
        ),
        SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF",
        ),
    ],
)


class NotebookCell:
    """Represents a notebook cell with its content and metadata."""

    def __init__(
        self,
        source: str,
        cell_type: str,
        outputs: Optional[List] = None,
        index: int = 0,
    ) -> None:
        self.source = source
        self.cell_type = cell_type
        self.outputs = outputs or []
        self.index = index

    def __repr__(self) -> str:
        return f"NotebookCell(index={self.index}, cell_type='{self.cell_type}')"


class NotebookStructure:
    """Holds notebook structure information."""

    def __init__(self) -> None:
        self.title: str = ""
        self.introduction: str = ""
        self.sections: List[Dict[str, Any]] = []


class NotebookProcessor:
    """Processes Jupyter notebooks to analyze structure and generate documentation."""

    def __init__(self, project_id: str, location: str) -> None:
        """Initializes the NotebookProcessor with Vertex AI credentials."""
        self.client = genai.Client(vertexai=True, project=project_id, location=location)

    def read_notebook(
        self, input_path: str
    ) -> Tuple[List[NotebookCell], Dict[str, str]]:
        """Reads a Jupyter notebook and extracts cells and content."""
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input notebook file not found: {input_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading notebook: {e}")

        cells: List[NotebookCell] = []
        code_parts: List[str] = []
        markdown_parts: List[str] = []

        for idx, cell in enumerate(nb.cells):
            if cell.source.strip():
                outputs = getattr(cell, "outputs", []) or []
                cells.append(
                    NotebookCell(
                        source=cell.source,
                        cell_type=cell.cell_type,
                        outputs=outputs,
                        index=idx,
                    )
                )
                if cell.cell_type == "code":
                    code_parts.append(cell.source)
                elif cell.cell_type == "markdown":
                    markdown_parts.append(cell.source)

        full_notebook_content = {
            "code": "\n\n".join(code_parts),
            "markdown": "\n\n".join(markdown_parts),
        }
        return cells, full_notebook_content

    def _prepare_code_cells_for_prompt(self, code_cells: List[NotebookCell]) -> str:
        """Prepares a JSON string of code cell information for the LLM prompt."""
        code_cells_for_prompt = [
            {
                "code_index": code_idx,
                "original_index": c.index,
                "content": c.source.strip(),
            }
            for code_idx, c in enumerate(code_cells)
        ]
        return json.dumps(code_cells_for_prompt, ensure_ascii=False, indent=2)

    def analyze_notebook_structure(
        self, cells: List[NotebookCell], full_content: Dict[str, str]
    ) -> NotebookStructure:
        """
        Analyzes only the code cells to determine structure (sections), title, and introduction.
        The LLM assigns sections using the code_cell indices, ignoring markdown cells.
        """
        code_cells = [cell for cell in cells if cell.cell_type == "code"]
        if not code_cells:
            default_structure = NotebookStructure()
            default_structure.title = "Python Notebook"
            return default_structure

        cells_json_str = self._prepare_code_cells_for_prompt(code_cells)

        prompt = f"""
        You are given a list of code cells from a Jupyter Notebook. Each item has:
        - 'code_index': the sequential index among code-only cells,
        - 'original_index': the cell's position in the full notebook (for reference),
        - 'content': the code content.

        **Goal**:
        1. Suggest a descriptive, technical title for the notebook.
        2. Generate an introduction paragraph that will be added just after the title. This introduction should briefly summarize the notebook's purpose and key topics.
        3. Divide these code cells into 2-10 logical sections based on functionality, context, and flow.
        4. For each section, provide:
        - A 'title' (subtitle),
        - A short 'description',
        - 'start_cell' and 'end_cell' (both inclusive), which refer to 'code_index' in the list below.

        **Your output** must follow this JSON schema exactly:
        {json.dumps(NOTEBOOK_STRUCTURE_SCHEMA, indent=2)}

        Here are the code cells:
        {cells_json_str}

        Additionally, here's the entire notebook's code and markdown for background context:
        - Markdown content:
        ```markdown
        {full_content['markdown']}
        ```

        - Code content:
        ```python
        {full_content['code']}
        ```

        Important:
        - The first code_index is 0, the last code_index is {len(code_cells) - 1}.
        - Use code_index in 'start_cell' and 'end_cell'; do not reference 'original_index'.
        - The final structure should remain consistent (start_cell <= end_cell).
        """

        try:
            config = GENERATION_CONFIG.model_copy()
            config.response_mime_type = "application/json"
            config.response_schema = NOTEBOOK_STRUCTURE_SCHEMA
            response = self.client.models.generate_content(
                model=MODEL_ID,
                config=config,
                contents=prompt,
            )
            structure_data: Dict = json.loads(response.text)
            return self._validate_and_create_structure(structure_data, len(code_cells))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON response from model: {e}")
        except Exception as e:
            print(f"Error analyzing notebook structure: {e}")
            return self._create_default_structure(len(code_cells))

    def _validate_and_create_structure(
        self, structure_data: Dict, num_code_cells: int
    ) -> NotebookStructure:
        """Validates the structure data from the LLM and creates a NotebookStructure object."""
        notebook_structure = NotebookStructure()
        notebook_structure.title = structure_data.get("title", "Untitled Notebook")
        notebook_structure.introduction = structure_data.get("introduction", "")
        notebook_structure.sections = []
        for section in structure_data.get("sections", []):
            start_cell = max(0, min(section["start_cell"], num_code_cells - 1))
            end_cell = max(0, min(section["end_cell"], num_code_cells - 1))
            section["start_cell"] = start_cell
            section["end_cell"] = max(start_cell, end_cell)  # Ensure start <= end
            notebook_structure.sections.append(section)

        # Ensure coverage of all code cells in the last section if needed
        if (
            notebook_structure.sections
            and notebook_structure.sections[-1]["end_cell"] < num_code_cells - 1
        ):
            notebook_structure.sections[-1]["end_cell"] = num_code_cells - 1

        return notebook_structure

    def _create_default_structure(self, num_code_cells: int) -> NotebookStructure:
        """Creates a default NotebookStructure when analysis fails."""
        default_structure = NotebookStructure()
        default_structure.title = "Python Notebook"
        if num_code_cells > 0:
            default_structure.sections = [
                {
                    "title": "Main Section",
                    "description": "Main notebook content",
                    "start_cell": 0,
                    "end_cell": num_code_cells - 1,
                }
            ]
        return default_structure

    def generate_section_markdown(
        self,
        cells: List[NotebookCell],
        section_info: Dict,
        full_content: Dict[str, str],
    ) -> str:
        """Generates markdown documentation for a section of the notebook."""
        code_cells = [cell for cell in cells if cell.cell_type == "code"]
        section_code_cells = code_cells[
            section_info["start_cell"] : section_info["end_cell"] + 1
        ]

        prompt = f"""
        Generate documentation for this section of the notebook.

        Context:
        1. Section: {section_info['title']}
        Description: {section_info['description']}

        2. Code in this section:
        ```python
        {chr(10).join(cell.source for cell in section_code_cells)}
        ```
        
        3. Complete notebook context:
        Markdown: ```markdown
        {full_content['markdown']}
        ```
        Code: ```python
        {full_content['code']}
        ```
        
        Requirements:
        1. Your text should exclusively talk about the Code of this section
        2. The text you are writing for this section should fit the overall code and overall initial markdown but still be exclusive to the code of the section.
        3. Focus on the purpose and outcomes rather than line-by-line explanation
        4. Explain concepts and approaches rather than just code functionality
        5. Maintain a narrative flow that connects to the overall notebook purpose
        6. Use appropriate markdown formatting
        7. Don't reference cell numbers or positions
        8. Group related operations together in the explanation
        9. Be concise, use structure and bullet points when necessary.
        10. Focus more on functional rather than code and syntax details.
        11. Keep it short and concise, very concise.
        
        Return formatted markdown content that creates a clear narrative for this section.
        """

        try:
            response = self.client.models.generate_content(
                model=MODEL_ID,
                config=GENERATION_CONFIG,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(
                f"Error generating section markdown for '{section_info['title']}': {e}"
            )
            return f"<!-- Error generating markdown: {e} -->\n\n### {section_info['title']}"

    def create_structured_notebook(
        self,
        cells: List[NotebookCell],
        structure: NotebookStructure,
        full_content: Dict[str, str],
    ) -> nbformat.NotebookNode:
        """Creates a new notebook with structured sections and markdown."""
        new_nb = new_notebook()

        # Add notebook title and introduction
        new_nb.cells.append(new_markdown_cell(f"# {structure.title}\n\n"))
        new_nb.cells.append(new_markdown_cell(f"{structure.introduction}\n\n"))

        # Add table of contents
        new_nb.cells.append(self._generate_table_of_contents(structure))

        # Filter only code cells
        code_cells = [cell for cell in cells if cell.cell_type == "code"]

        # Process sections
        with tqdm(total=len(structure.sections), desc="Processing sections") as pbar:
            for section_idx, section in enumerate(structure.sections):
                # Generate section documentation
                section_markdown = (
                    self.generate_section_markdown(cells, section, full_content)
                    .removeprefix("```markdown")
                    .removesuffix("```")
                )
                new_nb.cells.append(new_markdown_cell(section_markdown))

                # Add only relevant code cells in this section
                for cell_idx in range(section["start_cell"], section["end_cell"] + 1):
                    if cell_idx < len(code_cells):  # Ensure the index is within bounds
                        cell = code_cells[cell_idx]
                        new_code_cell_ = new_code_cell(cell.source)
                        new_code_cell_.outputs = cell.outputs
                        new_code_cell_.execution_count = None
                        new_nb.cells.append(new_code_cell_)

                # Add section separator if not last section
                if section_idx < len(structure.sections) - 1:
                    new_nb.cells.append(new_markdown_cell("\n---\n"))
                pbar.update(1)

        return new_nb

    def _generate_table_of_contents(
        self, structure: NotebookStructure
    ) -> nbformat.NotebookNode:
        """Generates a table of contents markdown cell."""
        toc = "## Table of Contents\n\n"
        for idx, section in enumerate(structure.sections, 1):
            safe_title = section["title"].lower().replace(" ", "-")
            toc += f"{idx}. [{section['title']}](#{safe_title})\n"
        return new_markdown_cell(toc)

    def enrich_notebook(self, input_path: str, output_path: str) -> None:
        """Enriches a notebook with structured markdown and sections."""
        try:
            # Read notebook
            cells, full_content = self.read_notebook(input_path)

            # Analyze structure
            structure = self.analyze_notebook_structure(cells, full_content)

            # Create structured notebook
            new_nb = self.create_structured_notebook(cells, structure, full_content)

            # Save enriched notebook
            with open(output_path, "w", encoding="utf-8") as f:
                nbformat.write(new_nb, f)

            print(f"Successfully enriched notebook: {output_path}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error processing notebook content: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Example usage
    project_id = "your-project-id"
    location = "us-central1"
    input_notebook_path = "test.ipynb"  # Replace with your input notebook path
    output_notebook_path = (
        "test_enriched.ipynb"  # Replace with your desired output path
    )

    processor = NotebookProcessor(project_id, location)
    processor.enrich_notebook(input_notebook_path, output_notebook_path)
