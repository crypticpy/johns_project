"""
Service Desk Onboarding Analyzer

This application provides advanced analysis for identifying department onboarding opportunities in ServiceNow.
Key features include:
- Dataset upload and management
- Ticket quality and complexity visualizations
- Department-level insights
- Comparison across multiple datasets
- Historical tracking of analyses
- Automated report generation with visualizations

Dependencies:
- streamlit
- sqlite3
- pandas
- plotly
- openai (AzureOpenAI SDK)
"""

# Standard library imports
import datetime
import hashlib
import io
import json
import sqlite3
from typing import Dict, List, Optional, Tuple

# Third-party imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ONBOARDING_GUIDELINES = """
# Service Desk Onboarding Framework

## Key Principles:
- **Tiering Heuristic**: If typical resolution > ~8 minutes, treat as Tier 2 or convert to catalog request
- **Self-service first**: Maximize Tier 0 (SSPR, KB, forms). Prioritize catalog creation for repeatables
- **Data-first discovery**: Analyze ServiceNow tickets by department to extract apps/resources, common issues, historical resolvers, fix patterns, and typical handle times

## Onboarding Flow:
1. **Pre-Intake**: Confirm scope, users, channels, domains/permissions, enterprise apps
2. **Data Discovery**: Run ServiceNow analyses; produce landscape
3. **Knowledge Plan**: Dept drafts KBs; SD maintains
4. **Tiering & Routing**: Define incident vs request; apply 8-min heuristic
5. **Enablement**: Ensure SSPR dynamic groups exist
6. **Metrics & Staffing**: Forecast volume × AHT
7. **Operationalize**: Publish checklist, KBs, catalog

## Red Flags:
- Poor ticket quality
- High reassignment counts
- Repetitive issues without KB articles
- High complexity tasks at Tier 1
- Single-owner admin knowledge (SPOF risk)
"""

def filter_by_departments(df: pd.DataFrame, departments: list) -> pd.DataFrame:
    """Filter a dataset DataFrame by a list of department names."""
    if not departments or 'Department' not in df.columns:
        return df
    return df[df['Department'].isin(departments)]


def init_database() -> None:
    """Initialize the SQLite database with required tables for analyses and datasets."""
    conn = sqlite3.connect('onboarding_analyses.db')
    cursor = conn.cursor()

    # Create analyses table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        dataset_name TEXT NOT NULL,
        dataset_hash TEXT NOT NULL,
        department_filter TEXT,
        question TEXT NOT NULL,
        analysis_result TEXT NOT NULL,
        ticket_count INTEGER,
        metrics TEXT
    )
    """)

    # Create datasets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        upload_timestamp TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_hash TEXT UNIQUE NOT NULL,
        row_count INTEGER,
        department_count INTEGER,
        metadata TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_analysis(dataset_name: str, dataset_hash: str, department_filter: Optional[str], question: str, analysis_result: str, ticket_count: int, metrics: dict) -> None:
    """Save an analysis record to the SQLite database."""
    conn = sqlite3.connect('onboarding_analyses.db')
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO analyses (timestamp, dataset_name, dataset_hash, department_filter, question, analysis_result, ticket_count, metrics)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(),
        dataset_name,
        dataset_hash,
        department_filter,
        question,
        analysis_result,
        ticket_count,
        json.dumps(metrics)
    ))
    conn.commit()
    conn.close()

def get_analysis_history(limit: int = 50) -> pd.DataFrame:
    """Retrieve historical analysis records from the database."""
    conn = sqlite3.connect('onboarding_analyses.db')
    query = f"SELECT * FROM analyses ORDER BY timestamp DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Initialize database at module level
init_database()

from typing import Dict, List, Optional, Tuple

class DatasetManager:
    def __init__(self) -> None:
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}

    def add_dataset(self, name: str, df: pd.DataFrame, file_hash: str) -> None:
        self.datasets[name] = df
        self.metadata[name] = {
            "hash": file_hash,
            "rows": len(df),
            "departments": df['Department'].nunique() if 'Department' in df.columns else 0,
            "upload_time": datetime.datetime.now().isoformat()
        }

    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        return self.datasets.get(name)

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())

    def get_comparison_stats(self) -> pd.DataFrame:
        stats: List[Dict] = []
        for name, df in self.datasets.items():
            stat_entry = {
                "Dataset": name,
                "Tickets": len(df),
                "Departments": df['Department'].nunique() if 'Department' in df.columns else 0,
                "Assignment Groups": df['Assignment Group'].nunique() if 'Assignment Group' in df.columns else 0
            }
            if 'ticket_quality' in df.columns:
                stat_entry["Avg Quality Score"] = self._quality_score(df['ticket_quality'])
            if 'Reassignment group count tracking_index' in df.columns:
                stat_entry["Avg Reassignments"] = df['Reassignment group count tracking_index'].mean()
            stats.append(stat_entry)
        return pd.DataFrame(stats)

    def _quality_score(self, quality_series: pd.Series) -> float:
        mapping = {
            "excellent": 5,
            "good": 4,
            "fair": 3,
            "poor": 2,
            "very poor": 1
        }
        return quality_series.map(mapping).mean()

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate MD5 hash for given file content."""
    return hashlib.md5(file_content).hexdigest()

class ChartGenerator:
    @staticmethod
    def create_quality_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
        if 'ticket_quality' not in df.columns:
            return None
        quality_counts = df['ticket_quality'].value_counts().reset_index()
        quality_counts.columns = ['Quality', 'Count']
        fig = px.pie(quality_counts, names='Quality', values='Count', title="Ticket Quality Distribution", color_discrete_sequence=px.colors.sequential.RdBu)
        return fig

    @staticmethod
    def create_complexity_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
        if 'resolution_complexity' not in df.columns:
            return None
        complexity_counts = df['resolution_complexity'].value_counts().reset_index()
        complexity_counts.columns = ['Complexity', 'Count']
        fig = px.bar(complexity_counts, x='Complexity', y='Count', title="Resolution Complexity Distribution", color='Count', color_continuous_scale='Viridis')
        return fig

    @staticmethod
    def create_department_volume(df: pd.DataFrame, top_n: int = 10) -> Optional[go.Figure]:
        if 'Department' not in df.columns:
            return None
        dept_counts = df['Department'].value_counts().nlargest(top_n).reset_index()
        dept_counts.columns = ['Department', 'Count']
        fig = px.bar(dept_counts, x='Count', y='Department', orientation='h', title=f"Top {top_n} Departments by Ticket Volume", color='Count', color_continuous_scale='Blues')
        return fig

    @staticmethod
    def create_reassignment_analysis(df: pd.DataFrame) -> Optional[go.Figure]:
        col_name = 'Reassignment group count tracking_index'
        if col_name not in df.columns:
            return None
        reassignment_counts = df[col_name].value_counts().sort_index().reset_index()
        reassignment_counts.columns = ['Reassignments', 'Count']
        fig = go.Figure(data=[go.Bar(x=reassignment_counts['Reassignments'], y=reassignment_counts['Count'], marker_color='indianred')])
        fig.update_layout(title="Reassignment Distribution", xaxis_title="Reassignments", yaxis_title="Count")
        return fig

    @staticmethod
    def create_product_distribution(df: pd.DataFrame, top_n: int = 15) -> Optional[go.Figure]:
        if 'extract_product' not in df.columns:
            return None
        product_counts = df['extract_product'].value_counts().nlargest(top_n).reset_index()
        product_counts.columns = ['Product/System', 'Count']
        fig = px.bar(product_counts, x='Count', y='Product/System', orientation='h', title=f"Top {top_n} Products/Systems by Ticket Volume", color='Count', color_continuous_scale='Oranges')
        return fig

    @staticmethod
    def create_comparison_chart(datasets: Dict[str, pd.DataFrame], metric: str) -> go.Figure:
        fig = go.Figure()
        for name, df in datasets.items():
            if metric == 'quality' and 'ticket_quality' in df.columns:
                counts = df['ticket_quality'].value_counts()
                fig.add_trace(go.Bar(name=name, x=counts.index, y=counts.values))
            elif metric == 'complexity' and 'resolution_complexity' in df.columns:
                counts = df['resolution_complexity'].value_counts()
                fig.add_trace(go.Bar(name=name, x=counts.index, y=counts.values))
        fig.update_layout(
            title=f"{metric.capitalize()} Comparison Across Datasets",
            barmode='group',
            xaxis_title=metric.capitalize(),
            yaxis_title='Count'
        )
        return fig

from openai import AzureOpenAI

class OnboardingAnalyzer:
    def __init__(self, azure_endpoint: str, api_key: str, deployment_name: str) -> None:
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview"
        )
        self.deployment_name = deployment_name

    def prepare_ticket_context(self, df: pd.DataFrame, max_tickets: int = 50, dataset_name: str = "Dataset") -> Tuple[str, Dict]:
        relevant_cols = [
            'Description', 'Assignment Group', 'Department', 'extract_product', 'summarize_ticket',
            'ticket_quality', 'information_completeness', 'resolution_complexity', 'historical_similarity',
            'Reassignment group count tracking_index'
        ]
        available_cols = [col for col in relevant_cols if col in df.columns]
        sample_df = df.sample(n=min(max_tickets, len(df)), random_state=42) if not df.empty else df

        context_parts = []
        metrics: Dict = {}

        # Overview
        context_parts.append(f"Dataset: {dataset_name}")
        context_parts.append(f"Total Tickets: {len(df)}")
        if 'Department' in df.columns:
            dept_count = df['Department'].nunique()
            context_parts.append(f"Departments: {dept_count}")
            metrics['department_count'] = dept_count
        if 'Assignment Group' in df.columns:
            ag_count = df['Assignment Group'].nunique()
            context_parts.append(f"Assignment Groups: {ag_count}")
            metrics['assignment_group_count'] = ag_count

        # Quality distribution
        if 'ticket_quality' in df.columns:
            quality_counts = df['ticket_quality'].value_counts().to_dict()
            context_parts.append(f"Ticket Quality Distribution: {quality_counts}")
            metrics['quality_distribution'] = quality_counts

        # Complexity distribution
        if 'resolution_complexity' in df.columns:
            complexity_counts = df['resolution_complexity'].value_counts().to_dict()
            context_parts.append(f"Resolution Complexity Distribution: {complexity_counts}")
            metrics['complexity_distribution'] = complexity_counts

        # Reassignment patterns
        if 'Reassignment group count tracking_index' in df.columns:
            reassignment_counts = df['Reassignment group count tracking_index'].value_counts().to_dict()
            context_parts.append(f"Reassignment Group Count: {reassignment_counts}")
            metrics['reassignment_distribution'] = reassignment_counts

        # Top departments
        if 'Department' in df.columns:
            top_departments = df['Department'].value_counts().nlargest(10).to_dict()
            context_parts.append(f"Top Departments: {top_departments}")
            metrics['top_departments'] = top_departments

        # Sample tickets
        context_parts.append("Sample Tickets:")
        for _, row in sample_df.iterrows():
            ticket_info = []
            for col in available_cols:
                ticket_info.append(f"{col}: {row[col]}")
            context_parts.append(" | ".join(ticket_info))

        return "\n".join(context_parts), metrics

    def analyze_onboarding_opportunities(self, ticket_context: str, user_question: str, comparison_mode: bool = False) -> str:
        system_prompt = (
            "You are an IT Service Management consultant specializing in Service Desk onboarding.\n"
            "Analyze ticket patterns to identify opportunities for improving onboarding processes.\n"
            f"{ONBOARDING_GUIDELINES}\n"
            "Look for patterns in routing, resolution, and ticket quality.\n"
            "CRITICAL: For catalog item design:\n"
            "- Identify required fields\n"
            "- Specify field types and validation rules\n"
            "- Suggest assignment group routing based on successful resolution patterns\n"
            "- Outline approval workflows\n"
            "- Recommend pre-population logic based on historical tickets\n"
        )
        if comparison_mode:
            system_prompt += "\nAdditionally, identify key differences between the datasets.\n"

        user_prompt = (
            f"Question: {user_question}\n\n"
            f"Ticket Context:\n{ticket_context}\n\n"
            "Expected Output Structure:\n"
            "Executive Summary\n"
            "Onboarding Opportunities\n"
            "Quick Wins\n"
            "Required Actions\n"
            "Risks & Considerations\n"
            "Metrics to Track\n"
        )
        if comparison_mode:
            user_prompt += "Comparison Insights\n"

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=16000
            )
            
            # DEBUG prints removed after identifying token budget issue
            if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
                content = getattr(response.choices[0].message, "content", None)
                return content if content else "No content returned from API"
            else:
                return "API returned empty choices array"
        except Exception as e:
            error_details = f"""
            Error calling Azure OpenAI: {str(e)}
            
            Error type: {type(e).__name__}
            
            Please check:
            1. Azure OpenAI endpoint is correct
            2. API key is valid
            3. Deployment name 'gpt-4o' exists
            4. Model has available capacity
            """
            return error_details

    def batch_analyze(self, ticket_context: str, questions: list, progress_callback: Optional[callable] = None) -> list:
        results = []
        for idx, question in enumerate(questions):
            if progress_callback:
                progress_callback(idx + 1, len(questions), question)
            analysis = self.analyze_onboarding_opportunities(ticket_context, question)
            results.append((question, analysis))
        return results

def main():
    # Page configuration
    st.set_page_config(
        page_title="Service Desk Onboarding Analyzer",
        page_icon="🎫",
        layout="wide"
    )

    # Main title and description
    st.title("Service Desk Onboarding Analyzer")
    st.markdown("""
    Advanced analysis tool for identifying department onboarding opportunities with multi-dataset comparison,
    historical tracking, and automated reporting capabilities.
    
    **Note:** Upload multiple datasets to compare different snapshots (e.g., different departments, time periods you've exported separately, or before/after intervention scenarios).
    """)
    st.write("A tool to help analyze onboarding tickets in ServiceNow.")

    # Initialize session state variables
    if 'dataset_manager' not in st.session_state:
        st.session_state['dataset_manager'] = DatasetManager()
    if 'batch_questions' not in st.session_state:
        st.session_state['batch_questions'] = []

    # Sidebar configuration section
    st.sidebar.header("⚙️ Azure AI Configuration")
    azure_endpoint = st.sidebar.text_input(
        "Azure OpenAI Endpoint",
        placeholder="https://your-resource.openai.azure.com/"
    )
    azure_api_key = st.sidebar.text_input(
        "API Key",
        type="password"
    )
    azure_deployment = st.sidebar.text_input(
        "Deployment Name",
        placeholder="gpt-4"
    )

    # Analysis Settings section after Azure config inputs
    st.sidebar.divider()
    st.sidebar.header("📊 Analysis Settings")
    max_tickets = st.sidebar.slider(
        "Max tickets to include in context",
        min_value=10,
        max_value=200,
        value=50,
        help="More tickets = more context but slower processing"
    )
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        options=["Single Dataset", "Comparison Mode", "Historical Review"],
        help="Choose your analysis approach",
        key="analysis_mode_radio"
    )
    st.sidebar.divider()
    with st.sidebar.expander("📋 View Onboarding Framework"):
        st.markdown(ONBOARDING_GUIDELINES)

    tabs = st.tabs([
        "📁 Data Upload",
        "🔍 Analysis",
        "📊 Visualizations",
        "📈 Historical Tracking",
        "📥 Reports"
    ])

    with tabs[0]:
        st.header("📁 Upload & Manage Datasets")
        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_files = st.file_uploader(
                "Upload Excel datasets (.xlsx)",
                type=["xlsx"],
                accept_multiple_files=True
            )
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    try:
                        file_content = uploaded_file.read()
                        file_hash = calculate_file_hash(file_content)
                        uploaded_file.seek(0)
                        df = pd.read_excel(io.BytesIO(file_content))
                        dataset_name = uploaded_file.name.replace(".xlsx", "")
                        st.session_state['dataset_manager'].add_dataset(dataset_name, df, file_hash)
                        st.success(f"Dataset '{dataset_name}' loaded successfully with {len(df)} tickets.")
                    except Exception as e:
                        st.error(f"Error loading file {uploaded_file.name}: {e}")

        with col2:
            st.subheader("Loaded Datasets")
            datasets = st.session_state['dataset_manager'].list_datasets()
            if datasets:
                for name in datasets:
                    df = st.session_state['dataset_manager'].get_dataset(name)
                    st.metric(label=name, value=len(df))
            else:
                st.info("No datasets loaded")

        if len(st.session_state['dataset_manager'].list_datasets()) > 1:
            st.divider()
            st.subheader("📊 Dataset Comparison")
            comparison_df = st.session_state['dataset_manager'].get_comparison_stats()
            st.dataframe(comparison_df, use_container_width=True)

    with tabs[1]:
        st.header("🔍 Analysis")
        
        datasets = st.session_state['dataset_manager'].list_datasets()
        if not datasets:
            st.warning("Please upload at least one dataset")
            return
        
        # Dataset selection based on analysis_mode
        if analysis_mode == "Single Dataset":
            selected_dataset_name = st.selectbox("Select Dataset", datasets, key="analysis_single_dataset_select")
            selected_datasets = [selected_dataset_name] if selected_dataset_name else []
        else:
            selected_datasets = st.multiselect("Select Datasets", datasets, default=datasets, key="analysis_multi_dataset_select")
        
        if not selected_datasets:
            st.warning("Please select at least one dataset")
            return
        
        # Department filtering
        st.subheader("🎯 Department Filtering")
        all_departments = set()
        for ds_name in selected_datasets:
            df = st.session_state['dataset_manager'].get_dataset(ds_name)
            if df is not None and 'Department' in df.columns:
                all_departments.update(df['Department'].dropna().unique())
        
        if all_departments:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_departments = st.multiselect(
                    "Filter by Department",
                    sorted(all_departments),
                    help="Select one or more departments to focus analysis",
                    key="analysis_department_filter"
                )
            with col2:
                st.metric("Total Departments", len(all_departments))
                if selected_departments:
                    st.metric("Selected Departments", len(selected_departments))
        else:
            st.info("No department information found")
            selected_departments = []
        
        st.divider()
        
        # PART 2: Questions interface
        st.subheader("💬 Questions")
        question_mode = st.radio(
            "Question Mode",
            options=["Single Question", "Batch Questions"],
            horizontal=True
        )
        user_question = ""

        if question_mode == "Single Question":
            example_questions = [
                "What departments should we prioritize for Service Desk onboarding?",
                "What catalog items or KB articles should we create based on this ticket data?",
                "Design a complete catalog item for [specific issue] with fields and routing",
                "Which issues are good candidates for Tier 1 vs Tier 2 handling?",
                "What are the biggest routing or triage problems we need to solve?",
                "What knowledge gaps exist that could cause single-point-of-failure risks?",
                "Custom question..."
            ]
            selected_question_option = st.selectbox("Select or write a question:", example_questions, key="analysis_single_question_select")
            if selected_question_option == "Custom question...":
                user_question = st.text_area("Your question:", height=100, key="custom_question_textarea")
            else:
                user_question = st.text_area("Your question:", value=selected_question_option, height=100, key="single_question_textarea")

        else:
            st.info("📋 Add multiple questions to analyze in sequence")
            new_batch_question = st.text_input("Add question to batch:")
            if st.button("➕ Add to Batch"):
                if new_batch_question.strip():
                    st.session_state['batch_questions'].append(new_batch_question.strip())
                    st.success(f"Added to batch. Total questions: {len(st.session_state['batch_questions'])}")
            if st.session_state['batch_questions']:
                st.subheader(f"Batch Questions ({len(st.session_state['batch_questions'])})")
                for idx, q in enumerate(st.session_state['batch_questions'], start=1):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"{idx}. {q}")
                    with col2:
                        if st.button(f"🗑️ Delete {idx}"):
                            st.session_state['batch_questions'].pop(idx-1)
                            st.experimental_rerun()
                if st.button("🧹 Clear Batch"):
                    st.session_state['batch_questions'] = []
                    st.success("Batch cleared")

        st.divider()

        # Analyze button
        analyze_label = "🚀 Analyze" if question_mode == "Single Question" else f"🚀 Run Batch Analysis ({len(st.session_state['batch_questions'])} questions)"
        analyze_disabled = not (azure_endpoint and azure_api_key and azure_deployment)
        
        if st.button(analyze_label, type="primary", disabled=analyze_disabled):
            # Validation
            if question_mode == "Single Question" and not user_question.strip():
                st.warning("Please enter a question")
                return
            if question_mode != "Single Question" and not st.session_state['batch_questions']:
                st.warning("Please add at least one question to the batch")
                return
            
            # Prepare filtered datasets
            filtered_datasets = {}
            for ds_name in selected_datasets:
                df = st.session_state['dataset_manager'].get_dataset(ds_name)
                filtered_datasets[ds_name] = filter_by_departments(df, selected_departments)
            
            # Initialize analyzer
            analyzer = OnboardingAnalyzer(azure_endpoint, azure_api_key, azure_deployment)
            
            if question_mode == "Single Question":
                st.spinner("Analyzing ticket data...")
                if len(filtered_datasets) == 1:
                    ds_name = list(filtered_datasets.keys())[0]
                    df = filtered_datasets[ds_name]
                    context, metrics = analyzer.prepare_ticket_context(df, max_tickets, ds_name)
                    analysis = analyzer.analyze_onboarding_opportunities(context, user_question)
                    save_analysis(ds_name, st.session_state['dataset_manager'].metadata[ds_name]['hash'],
                                  ",".join(selected_departments) if selected_departments else None,
                                  user_question, analysis, len(df), metrics)
                    st.success("✅ Analysis Complete!")
                    st.divider()
                    st.header("📋 Analysis Results")
                    st.markdown(analysis)
                    st.download_button("💾 Download Analysis", data=analysis, file_name="analysis.md")
                else:
                    # Comparison mode
                    combined_contexts = []
                    for ds_name, df in filtered_datasets.items():
                        context, metrics = analyzer.prepare_ticket_context(df, max_tickets, ds_name)
                        combined_contexts.append(context)
                    combined_context_str = "\n\n---\n\n".join(combined_contexts)
                    analysis = analyzer.analyze_onboarding_opportunities(combined_context_str, user_question, comparison_mode=True)
                    st.success("✅ Comparison Analysis Complete!")
                    st.divider()
                    st.header("📋 Analysis Results")
                    st.markdown(analysis)
                    st.download_button("💾 Download Analysis", data=analysis, file_name="comparison_analysis.md")
            else:
                # Batch mode
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_list = []
                total_questions = len(st.session_state['batch_questions'])
                
                for idx, question in enumerate(st.session_state['batch_questions'], start=1):
                    progress_bar.progress(idx / total_questions)
                    status_text.text(f"Analyzing question {idx}/{total_questions}: {question}")
                    for ds_name in selected_datasets:
                        df = filtered_datasets[ds_name]
                        context, metrics = analyzer.prepare_ticket_context(df, max_tickets, ds_name)
                        analysis = analyzer.analyze_onboarding_opportunities(context, question)
                        save_analysis(ds_name, st.session_state['dataset_manager'].metadata[ds_name]['hash'],
                                      ",".join(selected_departments) if selected_departments else None,
                                      question, analysis, len(df), metrics)
                        results_list.append((question, analysis))
                
                st.success("✅ Batch Analysis Complete!")
                for q, analysis in results_list:
                    with st.expander(f"📌 {q}"):
                        st.markdown(analysis)
                
                combined_report = "\n\n".join([f"## {q}\n\n{analysis}" for q, analysis in results_list])
                st.download_button("💾 Download Combined Report", data=combined_report, file_name="batch_analysis_report.md")
        

    with tabs[2]:
        st.header("📊 Visualizations")
        datasets = st.session_state['dataset_manager'].list_datasets()
        if not datasets:
            st.warning("Please upload at least one dataset to view visualizations.")
        else:
            selected_dataset_name = st.selectbox("Select Dataset for Visualization", datasets, key="visualization_dataset_select")
            selected_df = st.session_state['dataset_manager'].get_dataset(selected_dataset_name)
            if selected_df is not None:
                # Quality distribution
                fig_quality = ChartGenerator.create_quality_distribution(selected_df)
                if fig_quality:
                    st.plotly_chart(fig_quality, use_container_width=True)

                # Complexity and Reassignment charts
                col1, col2 = st.columns(2)
                with col1:
                    fig_complexity = ChartGenerator.create_complexity_distribution(selected_df)
                    if fig_complexity:
                        st.plotly_chart(fig_complexity, use_container_width=True)
                with col2:
                    fig_reassignment = ChartGenerator.create_reassignment_analysis(selected_df)
                    if fig_reassignment:
                        st.plotly_chart(fig_reassignment, use_container_width=True)

                # Department volume
                fig_dept = ChartGenerator.create_department_volume(selected_df, top_n=15)
                if fig_dept:
                    st.plotly_chart(fig_dept, use_container_width=True)

                # Product distribution
                fig_product = ChartGenerator.create_product_distribution(selected_df, top_n=15)
                if fig_product:
                    st.plotly_chart(fig_product, use_container_width=True)

                # Dataset comparison charts
                if len(datasets) > 1:
                    st.divider()
                    st.subheader("📊 Dataset Comparison Charts")
                    comparison_metric = st.selectbox("Select metric for comparison", ["quality", "complexity"], key="visualization_comparison_metric_select")
                    comparison_datasets = {name: st.session_state['dataset_manager'].get_dataset(name) for name in datasets}
                    fig_comparison = None
                    if comparison_metric == "quality":
                        # Create combined quality comparison chart
                        fig_comparison = go.Figure()
                        for name, df in comparison_datasets.items():
                            if 'ticket_quality' in df.columns:
                                counts = df['ticket_quality'].value_counts()
                                fig_comparison.add_trace(go.Bar(name=name, x=counts.index, y=counts.values))
                        fig_comparison.update_layout(
                            title="Quality Comparison Across Datasets",
                            barmode='group',
                            xaxis_title="Quality",
                            yaxis_title="Count"
                        )
                    elif comparison_metric == "complexity":
                        # Create combined complexity comparison chart
                        fig_comparison = go.Figure()
                        for name, df in comparison_datasets.items():
                            if 'resolution_complexity' in df.columns:
                                counts = df['resolution_complexity'].value_counts()
                                fig_comparison.add_trace(go.Bar(name=name, x=counts.index, y=counts.values))
                        fig_comparison.update_layout(
                            title="Complexity Comparison Across Datasets",
                            barmode='group',
                            xaxis_title="Complexity",
                            yaxis_title="Count"
                        )
                    if fig_comparison:
                        st.plotly_chart(fig_comparison, use_container_width=True)

    with tabs[3]:
        st.header("📈 Historical Tracking")
        st.info("Tab content coming soon")

    with tabs[4]:
        st.header("📥 Automated Report Generation")
        st.markdown("""
        Generate comprehensive onboarding reports combining automated analyses and visualizations.
        Select a dataset, choose key questions, and produce a downloadable markdown report ready for sharing.
        """)
        
        datasets = st.session_state['dataset_manager'].list_datasets()
        if not datasets:
            st.warning("Please upload at least one dataset to generate reports.")
            return
        
        selected_dataset_name = st.selectbox("Select Dataset for Report", datasets, key="reports_dataset_select")
        selected_df = st.session_state['dataset_manager'].get_dataset(selected_dataset_name)
        
        if selected_df is not None:
            report_questions = st.multiselect(
                "Select questions for report",
                options=[
                    "What departments should we prioritize for Service Desk onboarding?",
                    "What catalog items should we create with complete field specifications?",
                    "Design catalog items for top 3 repetitive issues with routing details",
                    "Which issues are good candidates for Tier 1 vs Tier 2 handling?",
                    "What are the biggest routing or triage problems?",
                    "What knowledge gaps exist that could cause SPOF risks?"
                ],
                default=[
                    "What departments should we prioritize for Service Desk onboarding?",
                    "What catalog items should we create with complete field specifications?"
                ],
                key="reports_questions_select"
            )
            
            if st.button("📄 Generate Report", type="primary"):
                if not (azure_endpoint and azure_api_key and azure_deployment):
                    st.error("Azure configuration is incomplete. Please set in the sidebar.")
                    return
                with st.spinner("Generating comprehensive report..."):
                    analyzer = OnboardingAnalyzer(azure_endpoint, azure_api_key, azure_deployment)
                    chart_gen = ChartGenerator()
                    
                    analyses = []
                    for question in report_questions:
                        context, metrics = analyzer.prepare_ticket_context(selected_df, max_tickets, selected_dataset_name)
                        analysis = analyzer.analyze_onboarding_opportunities(context, question)
                        analyses.append((question, analysis))
                    
                    charts = []
                    charts.append(chart_gen.create_quality_distribution(selected_df))
                    charts.append(chart_gen.create_complexity_distribution(selected_df))
                    charts.append(chart_gen.create_department_volume(selected_df))
                    charts.append(chart_gen.create_reassignment_analysis(selected_df))
                    charts.append(chart_gen.create_product_distribution(selected_df))
                    
                    report_md = generate_report(selected_dataset_name, selected_df, analyses, charts)
                    
                    st.success("✅ Report Generated!")
                    with st.expander("📄 Preview Report"):
                        st.markdown(report_md)
                    
                    st.subheader("📊 Report Visualizations")
                    for chart in charts:
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                    
                    st.download_button(
                        "💾 Download Complete Report",
                        data=report_md,
                        file_name=f"{selected_dataset_name}_report.md"
                    )

def generate_report(dataset_name: str, df: pd.DataFrame, analyses: list, charts: list) -> str:
    md = f"# Automated Onboarding Report\n\n"
    md += f"**Dataset:** {dataset_name}\n\n"
    md += f"**Generated On:** {datetime.datetime.now().isoformat()}\n\n"
    md += f"## Dataset Summary\n"
    md += f"- Total Tickets: {len(df)}\n"
    if 'Department' in df.columns:
        md += f"- Departments: {df['Department'].nunique()}\n"
    if 'Assignment Group' in df.columns:
        md += f"- Assignment Groups: {df['Assignment Group'].nunique()}\n"
    md += "\n"
    
    for question, analysis in analyses:
        md += f"## {question}\n\n"
        md += f"{analysis}\n\n"
    
    md += f"## Charts Included: {len([c for c in charts if c is not None])}\n"
    return md

if __name__ == "__main__":
    main()
