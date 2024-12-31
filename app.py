import streamlit as st
import os
import tempfile
import nbformat
from nbconvert import HTMLExporter
import json
import streamlit.components.v1 as components

from notebook_enricher import NotebookProcessor

def initialize_notebook_processor(project_id: str, location: str = "us-central1"):
    """
    Initialize Vertex AI with project settings and return a NotebookProcessor instance.
    
    Args:
        project_id: GCP project ID
        location: GCP region (default: us-central1)
    
    Returns:
        NotebookProcessor instance if successful, None if failed
    """
    try:
        processor = NotebookProcessor(project_id=project_id, location=location)
        return processor
    except Exception as e:
        st.error(f"Failed to initialize NotebookProcessor: {str(e)}")
        
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            st.error("""
                Missing Google Cloud credentials. Please ensure:
                1. You have a service account key file
                2. The GOOGLE_APPLICATION_CREDENTIALS environment variable is set
                3. The service account has necessary permissions
            """)
        if not project_id:
            st.error("Project ID is required")
        
        return None

def convert_notebook_to_html(notebook_content):
    """Convert notebook content to HTML for rendering."""
    try:
        # Load notebook
        if isinstance(notebook_content, str):
            # nb_dict = json.loads(notebook_content)
            nb_dict = nbformat.reads(notebook_content, as_version=4)
        else:
            # nb_dict = json.loads(notebook_content.decode())
            nb_dict = nbformat.reads(notebook_content.decode(), as_version=4)
        
        # Fix source content if it's a list
        for cell in nb_dict.get('cells', []):
            if isinstance(cell.get('source', ''), list):
                cell['source'] = ''.join(cell['source'])
            if cell['cell_type'] == 'code' and 'outputs' not in cell:
                cell['outputs'] = []
        
        # Create valid notebook node
        nb = nbformat.from_dict(nb_dict)
        
        # Configure HTML exporter
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        
        # Convert notebook to HTML
        (body, _) = html_exporter.from_notebook_node(nb)
        
        # Add custom CSS for better Streamlit integration
        custom_css = """
        <style>
            .jupyter-notebook {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .input_area {
                background-color: #f8f9fa !important;
                border-radius: 4px;
                padding: 10px;
                margin: 5px 0;
            }
            .output_area {
                padding: 10px;
                margin: 5px 0;
            }
            .cell {
                margin: 20px 0;
            }
        </style>
        """
        
        return custom_css + body
        
    except Exception as e:
        st.error(f"Error converting notebook to HTML: {str(e)}")
        st.error("Falling back to JSON view")
        if isinstance(notebook_content, bytes):
            return f"<pre>{notebook_content.decode()}</pre>"
        return f"<pre>{notebook_content}</pre>"

def process_notebook(uploaded_file):
    """Process the uploaded notebook file by enriching it with AI-generated markdown."""
    try:
        # Create temp files for input and output
        with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as input_temp:
            input_temp.write(uploaded_file.getvalue())
            input_path = input_temp.name
            
        output_path = input_path.replace('.ipynb', '_enriched.ipynb')
        
        # Initialize the NotebookProcessor and process notebook
        processor = initialize_notebook_processor(project_id=st.session_state.project_id)
        if processor:
            with st.spinner('Enriching notebook...'):
                processor.enrich_notebook(input_path, output_path)
            
            # Read the enriched notebook
            with open(output_path, 'r', encoding='utf-8') as f:
                enriched_content = f.read()
                
            # Cleanup temp files
            os.unlink(input_path)
            os.unlink(output_path)
            
            return enriched_content
            
    except Exception as e:
        st.error(f"Error processing notebook: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Notebook Enricher Demo",
        page_icon="📚",
        layout="wide"
    )
    
    # Title and description
    st.title("📚 Notebook Enricher Demo")
    st.write("""
    Transform your Jupyter notebooks with AI-powered documentation and structure.
    Upload a notebook to see it enriched with:
    - Clear section organization
    - Detailed markdown explanations
    - Code documentation
    - Table of contents
    """)
    
    # Project ID input
    if 'project_id' not in st.session_state:
        st.session_state.project_id = ''
        
    with st.sidebar:
        st.header("Configuration")
        project_id = st.text_input(
            "GCP Project ID",
            value=st.session_state.project_id,
            help="Enter your Google Cloud Project ID"
        )
        st.session_state.project_id = project_id

    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Notebook")
        uploaded_file = st.file_uploader(
            "Upload a Jupyter notebook",
            type=['ipynb'],
            help="Select a .ipynb file to enrich"
        )
        
        if uploaded_file:
            # Convert and display original notebook
            html_content = convert_notebook_to_html(uploaded_file.getvalue())
            if html_content:
                components.html(html_content, height=600, scrolling=True)
            
            if st.button("🚀 Enrich Notebook", type="primary"):
                if not st.session_state.project_id:
                    st.error("Please enter your GCP Project ID in the sidebar")
                else:
                    enriched_content = process_notebook(uploaded_file)
                    if enriched_content:
                        st.session_state.enriched_content = enriched_content
                        st.success("✨ Notebook successfully enriched!")
                    
    with col2:
        st.header("Enriched Notebook")
        if 'enriched_content' in st.session_state:
            # Convert and display enriched notebook
            html_content = convert_notebook_to_html(st.session_state.enriched_content)
            if html_content:
                components.html(html_content, height=600, scrolling=True)
            
            # Download button
            st.download_button(
                label="📥 Download Enriched Notebook",
                data=st.session_state.enriched_content,
                file_name=f"enriched_{uploaded_file.name}",
                mime="application/x-ipynb+json",
            )

    # Feature highlights
    if uploaded_file and 'enriched_content' in st.session_state:
        st.header("✨ Enrichments Added")
        col1, col2, col3 = st.columns(3)
        
        # Parse notebooks to get actual metrics
        original_nb = nbformat.reads(uploaded_file.getvalue().decode(), as_version=4)
        enriched_nb = nbformat.reads(st.session_state.enriched_content, as_version=4)
        
        with col1:
            # Count markdown cells that contain section headers
            section_count = sum(1 for cell in enriched_nb.cells 
                                if cell.cell_type == 'markdown' and cell.source.strip().startswith('#'))
            st.metric("Sections Created", str(section_count))
        with col2:
            # Count difference in markdown cells
            original_markdown = sum(1 for cell in original_nb.cells if cell.cell_type == 'markdown')
            enriched_markdown = sum(1 for cell in enriched_nb.cells if cell.cell_type == 'markdown')
            st.metric("Markdown Cells Added", str(enriched_markdown - original_markdown))
        with col3:
            # Count code cells with new markdown above them (simple heuristic)
            documented_code = 0
            for i, cell in enumerate(enriched_nb.cells):
                if cell.cell_type == 'code' and i > 0:
                    if enriched_nb.cells[i - 1].cell_type == 'markdown':
                        documented_code += 1
            st.metric("Code Cells Documented", str(documented_code))

if __name__ == "__main__":
    main()
