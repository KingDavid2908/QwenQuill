import streamlit as st
import os
from dotenv import load_dotenv
import io
import datetime
import re
import random
import string
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
import operator
import docx
import fitz
from PIL import Image
import pytesseract
from langchain_community.utilities import SerpAPIWrapper
import assemblyai as aai

# Application theme configuration
APP_THEME = {
    'background_color': "#35A3A0",
    'button_bg_color': "#9BBA29", 
    'button_text_light': "#FCFEF9",
    'button_text_dark': "#000D0B",
    'text_area_bg': "#FCFEF9",
    'text_area_text': "#000D0B",
    'app_text_light': "#FCFEF9"
}

EDITOR_PLACEHOLDER = (
    "Welcome to Qwen Quill!\n\nPaste your text here, upload a document, or try out a feature like "
    "'Comprehensive Polish' or 'Summarize Text' from the sidebar.\n\nThis tool helps you refine your writing with the power of AI."
)

# Session state initialization
def initialize_session_state():
    """Initialize all session state variables with default values"""
    defaults = {
        'app_logs': [],
        'thread_id': f"thread_{os.urandom(8).hex()}",
        'transformation_history': [],
        'current_feature': "Comprehensive Polish",
        'user_text_area_value': EDITOR_PLACEHOLDER,
        'custom_transform_instruction': "",
        'file_processing_status': None,
        'last_processed_file_id': None,
        'ai_detection_result': None,
        'plagiarism_check_result': None,
        'audio_input_processed_id': None,
        'widget_key_suffix': 0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def create_log_entry(message, level="INFO"):
    """Create and store log entries with timestamp"""
    log_entry = f"[{level}] {datetime.datetime.now().strftime('%H:%M:%S')} {message}"
    print(log_entry)
    st.session_state.app_logs.append(log_entry)
    if len(st.session_state.app_logs) > 100:
        st.session_state.app_logs = st.session_state.app_logs[-100:]

def load_environment_variables():
    """Load and validate environment variables"""
    load_dotenv()
    
    config = {
        'openrouter_key': os.getenv("OPENROUTER_API_KEY"),
        'serpapi_key': os.getenv("SERPAPI_API_KEY"),
        'assemblyai_key': os.getenv("ASSEMBLYAI_API_KEY"),
        'site_url': os.getenv("YOUR_SITE_URL", "http://localhost:8501"),
        'site_name': os.getenv("YOUR_SITE_NAME", "VectronixAI_QwenQuill")
    }
    
    return config

def setup_external_services(config):
    """Configure external API services"""
    if config['assemblyai_key']:
        aai.settings.api_key = config['assemblyai_key']
        create_log_entry("AssemblyAI API key configured.")
    else:
        create_log_entry("AssemblyAI API key not found. Voice transcription will be disabled.", level="WARN")
    
    try:
        pytesseract.get_tesseract_version()
        create_log_entry("Tesseract OCR found and configured.")
    except Exception as e:
        create_log_entry(f"Tesseract OCR not found or not configured: {e}. Image OCR will not work.", level="ERROR")

def configure_page_styling():
    """Apply custom CSS styling to the Streamlit app"""
    theme = APP_THEME
    
    page_style = f"""
    <style>
        .stApp {{
            background-color: {theme['background_color']}; 
            color: {theme['app_text_light']};
        }}
        [data-testid="stHeader"] {{
            background-color: {theme['background_color']};
        }}
        [data-testid="stAppViewContainer"] > div {{
            background-color: {theme['background_color']};
        }}
        body {{
            background-color: {theme['background_color']};
        }}
        [data-testid="stSidebar"] > div:first-child {{ 
            background-color: {theme['background_color']};
            border-right: 1px solid #FFFFFF40;
        }}
        [data-testid="stSidebar"] [data-testid="stTabs"] label, 
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stTextInput > label,
        [data-testid="stSidebar"] .stSelectbox > label,
        [data-testid="stSidebar"] .stExpander > summary {{
            color: {theme['app_text_light']} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stTabs"] button {{
            color: {theme['app_text_light']}; 
        }}
        [data-testid="stSidebar"] [data-testid="stTabs"] button[aria-selected="true"] {{
            color: {theme['app_text_light']}; 
            font-weight: bold;
            border-bottom: 3px solid {theme['button_bg_color']};
        }}
        .stButton > button {{
            border: none;
            border-radius: 5px; 
            color: {theme['button_text_light']};
            background-color: {theme['button_bg_color']}; 
            font-weight: bold; 
            padding: 0.6em 1.2em;
            margin: 0.2em;
        }}
        div.stButton > button:hover {{
            filter: brightness(110%);
            color: {theme['button_text_light']};
        }}
        .stTextArea textarea {{
            background-color: {theme['text_area_bg']}; 
            color: {theme['text_area_text']}; 
            border: 1px solid {theme['button_bg_color']};
            border-radius: 5px;
            font-size: 1.1em;
        }}
        .stTextInput input {{
            background-color: {theme['text_area_bg']}CC;
            color: {theme['text_area_text']};
            border: 1px solid {theme['button_bg_color']};
        }}
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ 
            color: {theme['app_text_light']}; 
        }}
        .stCaption {{ 
            color: {theme['app_text_light']}99;
        }}
        .disabled-voice-button {{
            border: none; border-radius: 5px; color: {theme['button_text_light']}99;
            background-color: {theme['button_bg_color']}99;
            font-weight: bold; padding: 0.6em 1.2em; margin: 0.2em; width: 90%; display: block; text-align: center;
        }}
    </style>
    """
    st.markdown(page_style, unsafe_allow_html=True)

class AgentState(TypedDict):
    """State definition for the LangGraph agent"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    input_text_for_llm: str | None
    feature_args: dict | None
    search_queries: list[str] | None
    search_results: str | None

class PromptTemplates:
    """Central location for all system prompts"""
    
    BASE_INSTRUCTION = (
        "Unless the feature specifically asks for analysis or justification (like AI Content Detection or Plagiarism Check), "
        "your output should ONLY be the directly processed text without any of your own conversational phrases, "
        "self-commentary, introductions, or added titles/headers not present in the original. "
        "Adhere strictly to returning only the core requested output."
    )
    
    @classmethod
    def get_system_prompt(cls, feature_name, feature_args=None):
        """Generate appropriate system prompt based on feature type"""
        prompts = {
            "Comprehensive Polish": (
                "You are an expert editor. Your task is to meticulously polish the following text. "
                "Focus on improving clarity, grammar, spelling, punctuation, flow, sentence structure, and word choice for maximum impact. "
                "Preserve the original paragraph structure and intent. If paragraphs can be merged or split for better readability, do so. "
                "Ensure consistent tone and style. "
                f"{cls.BASE_INSTRUCTION}"
            ),
            "Translate Text": cls._get_translation_prompt(feature_args),
            "AI Content Detection": (
                "Analyze the provided text and estimate the likelihood that it was written by an AI. "
                "Provide your answer formatted EXACTLY as: [Percentage]%. Justification: [Your brief justification, typically one sentence]. "
                "For example: 75%. Justification: The text exhibits highly structured sentences and a neutral tone often seen in AI generation. "
                "Return ONLY this formatted string."
            ),
            "Summarize Text": (
                "You are an expert at summarizing text. Provide a concise summary of the following text, capturing the main points. "
                f"{cls.BASE_INSTRUCTION}"
            ),
            "Advanced Transformations": cls._get_transformation_prompt(feature_args),
            "Plagiarism Check (Simplified)": cls._get_plagiarism_prompt()
        }
        
        return prompts.get(feature_name, f"You are a helpful AI assistant. {cls.BASE_INSTRUCTION}")
    
    @classmethod
    def _get_translation_prompt(cls, feature_args):
        if not feature_args:
            return f"Translate the provided text. {cls.BASE_INSTRUCTION}"
        
        source_lang = feature_args.get('source_language', 'English')
        target_lang = feature_args.get('target_language', 'the specified target language')
        
        return (
            f"Translate the provided text from {source_lang} to {target_lang}. "
            f"If you do not support one or both of these languages, clearly state ONLY that you cannot perform the translation "
            f"for the specified languages and briefly explain why if possible. "
            f"Otherwise, provide ONLY the translated text. {cls.BASE_INSTRUCTION}"
        )
    
    @classmethod
    def _get_transformation_prompt(cls, feature_args):
        if not feature_args:
            return f"Improve the text generally. {cls.BASE_INSTRUCTION}"
        
        custom_instruction = feature_args.get("custom_instruction", "improve it generally")
        return (
            f"Take the following text and apply this instruction: '{custom_instruction}'. "
            f"{cls.BASE_INSTRUCTION}"
        )
    
    @classmethod
    def _get_plagiarism_prompt(cls):
        return (
            "You are an academic integrity assistant. Analyze the following main text for potential plagiarism based ONLY on the provided web search snippets. Do not perform any external searches. "
            "Identify if any snippets show significant textual overlap or strong conceptual similarity with the main text. "
            "Format your response EXACTLY as: Overall Similarity: [Low/Medium/High/Very High]. Matched Snippets: [List any source URLs and a brief quote from the snippet if highly similar, or 'None found.']. Justification: [Brief overall explanation for your assessment, mentioning specific overlaps if any]. "
            "Example: Overall Similarity: Medium. Matched Snippets: [https://example.com - \"The quick brown fox...\"]. Justification: The phrase 'The quick brown fox' is identical. Other parts differ."
            "Return ONLY this formatted string."
        )

class LLMProcessor:
    """Handles all LLM interactions and processing"""
    
    def __init__(self, api_key, site_url, site_name):
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.model_name = "qwen/qwen3-32b"
    
    def create_llm_client(self):
        """Create and configure LLM client"""
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": self.site_url, "X-Title": self.site_name},
            temperature=0.7,
            max_tokens=3000,
            request_timeout=120
        )
    
    def process_request(self, messages):
        """Process LLM request with error handling"""
        if not self.api_key:
            create_log_entry("OpenRouter API key not configured.", level="ERROR")
            return "Error: OpenRouter API key not configured."
        
        if not messages or not isinstance(messages[-1], HumanMessage):
            create_log_entry(f"Invalid message sequence for LLM. Last: {messages[-1] if messages else 'None'}", level="ERROR")
            return "Internal error: Invalid message sequence for LLM."
        
        try:
            llm_client = self.create_llm_client()
            create_log_entry(f"Sending to LLM ({self.model_name}): Sys: '{messages[0].content[:50]}' User: '{messages[-1].content[:50]}'")
            response = llm_client.invoke(messages)
            create_log_entry(f"LLM Response (100char): {response.content[:100]}")
            return response.content
        except Exception as e:
            create_log_entry(f"Error calling LLM: {e}", level="ERROR")
            error_message = f"LLM Error: {str(e)}"
            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                try:
                    error_details = e.response.json()
                    error_message += f" Details: {error_details.get('error', {}).get('message', '')}"
                except:
                    pass
            return error_message

class SearchService:
    """Handles web search functionality for plagiarism detection"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.search_tool = SerpAPIWrapper(serpapi_api_key=api_key) if api_key else None
    
    def generate_search_queries(self, text, llm_processor):
        """Generate search queries from input text"""
        if not text:
            return []
        
        query_gen_messages = [
            SystemMessage(content="You are a search query generation assistant. Given the following text, extract 2-3 distinct key phrases or concepts that would be effective search queries to find similar content online. Return ONLY the queries, each on a new line."),
            HumanMessage(content=text)
        ]
        
        try:
            response = llm_processor.process_request(query_gen_messages)
            queries = [q.strip() for q in response.split("\n") if q.strip()]
            create_log_entry(f"Generated search queries: {queries}")
            return queries[:3]
        except Exception as e:
            create_log_entry(f"Error generating search queries: {e}", level="ERROR")
            return []
    
    def perform_web_search(self, queries):
        """Execute web search for given queries"""
        if not queries or not self.search_tool:
            create_log_entry("No queries or SerpApi key for web search.", level="WARN")
            return "Web search could not be performed (no queries or API key)."
        
        all_snippets = []
        for query in queries:
            try:
                create_log_entry(f"Searching web for: {query}")
                results = self.search_tool.results(query)
                if "organic_results" in results:
                    for res in results["organic_results"][:2]:
                        snippet = res.get("snippet", "")
                        link = res.get("link", "")
                        title = res.get("title", "")
                        if snippet and link:
                            all_snippets.append(f"Source: {link}\nTitle: {title}\nSnippet: {snippet}\n---")
            except Exception as e:
                create_log_entry(f"Error during SerpApi search for query '{query}': {e}", level="ERROR")
                all_snippets.append(f"Error searching for '{query}'.\n---")
        
        return "\n".join(all_snippets) if all_snippets else "No relevant snippets found from web search."

# Graph node implementations
def prepare_llm_input(state: AgentState) -> dict:
    """Prepare input for LLM processing"""
    create_log_entry("--- Node: prepare_llm_input ---")
    text_to_process = state.get('input_text_for_llm', "")
    feature_args = state.get('feature_args', {})
    current_feature_name = feature_args.get("current_feature_name", "Comprehensive Polish")

    if not text_to_process and current_feature_name != "Plagiarism Check (Simplified)":
        return {"messages": [AIMessage(content="[NO_INPUT_PROVIDED_ERROR]")]}

    system_prompt_content = PromptTemplates.get_system_prompt(current_feature_name, feature_args)
    
    if current_feature_name == "Plagiarism Check (Simplified)":
        search_snippets = state.get("search_results", "No web search results provided.")
        text_to_process = f"Main Text to Check:\n{text_to_process}\n\nProvided Web Search Snippets for Comparison:\n{search_snippets}"

    constructed_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=text_to_process)
    ]
    return {"messages": constructed_messages, "search_queries": None, "search_results": None}

def generate_search_queries_for_plagiarism(state: AgentState):
    """Generate search queries for plagiarism detection"""
    create_log_entry("--- Node: generate_search_queries_for_plagiarism ---")
    text_to_check = state.get('input_text_for_llm', "")
    
    if not text_to_check:
        return {"search_queries": []}
    
    config = load_environment_variables()
    llm_processor = LLMProcessor(config['openrouter_key'], config['site_url'], config['site_name'])
    search_service = SearchService(config['serpapi_key'])
    
    queries = search_service.generate_search_queries(text_to_check, llm_processor)
    return {"search_queries": queries}

def perform_web_search_for_plagiarism(state: AgentState):
    """Execute web search for plagiarism detection"""
    create_log_entry("--- Node: perform_web_search_for_plagiarism ---")
    queries = state.get('search_queries', [])
    
    config = load_environment_variables()
    search_service = SearchService(config['serpapi_key'])
    search_results = search_service.perform_web_search(queries)
    
    return {"search_results": search_results}

def call_qwen_llm(state: AgentState):
    """Main LLM processing node"""
    create_log_entry("--- Node: call_qwen_llm ---")
    llm_messages = state['messages']

    if llm_messages and isinstance(llm_messages[0], AIMessage) and llm_messages[0].content == "[NO_INPUT_PROVIDED_ERROR]":
        return {"messages": [AIMessage(content="No text was provided to process. Please enter or upload text.")]}

    config = load_environment_variables()
    llm_processor = LLMProcessor(config['openrouter_key'], config['site_url'], config['site_name'])
    response_content = llm_processor.process_request(llm_messages)
    
    return {"messages": state["messages"] + [AIMessage(content=response_content)]}

def create_workflow_graph():
    """Create and configure the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    workflow.add_node("prepare_input", prepare_llm_input)
    workflow.add_node("generate_queries", generate_search_queries_for_plagiarism)
    workflow.add_node("web_search", perform_web_search_for_plagiarism)
    workflow.add_node("llm_call", call_qwen_llm)

    def should_route_to_plagiarism_flow(state: AgentState):
        feature_args = state.get('feature_args', {})
        if feature_args.get("current_feature_name") == "Plagiarism Check (Simplified)":
            return "generate_queries"
        return "prepare_input"

    workflow.set_conditional_entry_point(
        should_route_to_plagiarism_flow,
        {
            "generate_queries": "generate_queries",
            "prepare_input": "prepare_input"
        }
    )

    workflow.add_edge("generate_queries", "web_search")
    workflow.add_edge("web_search", "prepare_input")
    workflow.add_edge("prepare_input", "llm_call")
    workflow.add_edge("llm_call", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

class FileProcessor:
    """Handles file upload and processing operations"""
    
    @staticmethod
    def process_text_file(file_bytes):
        """Process plain text files with encoding fallback"""
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            create_log_entry("UTF-8 decode failed, trying latin-1.", level="WARN")
            return file_bytes.decode("latin-1", errors="replace")
    
    @staticmethod
    def process_docx_file(file_bytes):
        """Process Microsoft Word documents"""
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    
    @staticmethod
    def process_pdf_file(file_bytes):
        """Process PDF documents"""
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        content = "".join([page.get_text() for page in pdf_doc])
        pdf_doc.close()
        return content
    
    @staticmethod
    def process_image_file(file_bytes, filename):
        """Process image files using OCR"""
        try:
            image = Image.open(io.BytesIO(file_bytes))
            content = pytesseract.image_to_string(image)
            if not content.strip():
                create_log_entry(f"OCR for '{filename}' yielded no text.", level="WARN")
                return "", f"No text could be extracted from image '{filename}'."
            else:
                create_log_entry(f"OCR successful for '{filename}'.")
                return content, None
        except Exception as ocr_err:
            create_log_entry(f"Pytesseract OCR error for '{filename}': {ocr_err}", level="ERROR")
            return "", f"OCR failed for '{filename}'. Is Tesseract installed and in PATH?"

class AudioProcessor:
    """Handles voice input processing using AssemblyAI"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.is_available = bool(api_key)
    
    def transcribe_audio(self, audio_bytes, file_id):
        """Transcribe audio data using AssemblyAI"""
        if not self.is_available:
            return "", "AssemblyAI API key not configured"
        
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        temp_audio_path = f"temp_audio_stt_{random_suffix}.wav"
        transcribed_text = ""
        error_message = None
        
        try:
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)
            create_log_entry(f"Audio bytes from st.audio_input saved to {temp_audio_path}")

            if not os.path.exists(temp_audio_path):
                raise FileNotFoundError(f"Temporary audio file for STT not created: {temp_audio_path}")

            config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
            transcriber = aai.Transcriber(config=config)
            
            create_log_entry(f"Sending {temp_audio_path} to AssemblyAI...")
            transcript = transcriber.transcribe(temp_audio_path)

            if transcript.status == aai.TranscriptStatus.error:
                create_log_entry(f"AssemblyAI transcription failed: {transcript.error}", level="ERROR")
                error_message = f"Voice transcription error: {transcript.error}"
            elif transcript.text:
                transcribed_text = transcript.text
                create_log_entry(f"AssemblyAI transcription successful: {transcribed_text[:70]}...")
            else:
                create_log_entry("AssemblyAI transcription was empty but no error status.", level="WARN")
                error_message = "Voice transcription returned no text."

        except Exception as e:
            create_log_entry(f"Voice STT processing exception (AssemblyAI): {e}", level="ERROR")
            error_message = f"Could not transcribe voice: {e}"
        finally:
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    create_log_entry(f"Temp audio {temp_audio_path} removed.")
                except Exception as e_del:
                    create_log_entry(f"Error deleting temp audio {temp_audio_path}: {e_del}", level="WARN")
        
        return transcribed_text, error_message

class ResultProcessor:
    """Handles processing and parsing of AI results"""
    
    @staticmethod
    def process_ai_detection_result(ai_response):
        """Parse AI content detection response"""
        try:
            match = re.match(r"\[?(\d+)%\]?\.\s*Justification:\s*(.*)", ai_response, re.IGNORECASE)
            if match:
                percentage = int(match.group(1))
                justification = match.group(2)
                return {
                    "percentage": percentage,
                    "percentage_float": percentage / 100.0,
                    "justification": justification,
                    "raw_output": ai_response
                }
            else:
                return {
                    "raw_output": ai_response,
                    "percentage": 0,
                    "percentage_float": 0.0,
                    "justification": "Could not parse AI analysis."
                }
        except Exception as parse_ex:
            create_log_entry(f"Exception parsing AI Detection output: {parse_ex}", level="ERROR")
            return {
                "raw_output": ai_response,
                "percentage": 0,
                "percentage_float": 0.0,
                "justification": f"Error parsing AI analysis: {parse_ex}"
            }
    
    @staticmethod
    def process_plagiarism_result(ai_response):
        """Parse plagiarism check response"""
        overall_similarity_match = re.search(r"Overall Similarity:\s*\[([^\]\.]+)\]", ai_response, re.IGNORECASE)
        matched_snippets_match = re.search(r"Matched Snippets:\s*(.*?)\s*Justification:", ai_response, re.IGNORECASE | re.DOTALL)
        justification_match = re.search(r"Justification:\s*(.*)", ai_response, re.IGNORECASE | re.DOTALL)
        
        parsed_result = {
            "overall_similarity": overall_similarity_match.group(1).strip() if overall_similarity_match else "N/A",
            "matched_snippets": matched_snippets_match.group(1).strip() if matched_snippets_match else "None found.",
            "justification": justification_match.group(1).strip() if justification_match else "N/A",
            "raw_output": ai_response
        }
        
        similarity_mapping = {"Low": 0.25, "Medium": 0.5, "High": 0.75, "Very High": 0.9}
        parsed_result["percentage_float"] = similarity_mapping.get(parsed_result["overall_similarity"], 0.0)
        
        return parsed_result

class UIComponents:
    """Handles UI component creation and management"""
    
    @staticmethod
    def create_sidebar_tabs():
        """Create and populate sidebar tabs"""
        tab1, tab2, tab3 = st.tabs(["✨ Features", "📜 History", "📊 App Logs"])
        
        with tab1:
            UIComponents._create_features_tab()
        
        with tab2:
            UIComponents._create_history_tab()
        
        with tab3:
            UIComponents._create_logs_tab()
    
    @staticmethod
    def _create_features_tab():
        """Create features selection tab"""
        st.header("✨ Features")
        feature_options = (
            "Comprehensive Polish",
            "Translate Text",
            "AI Content Detection", 
            "Summarize Text",
            "Advanced Transformations",
            "Plagiarism Check (Simplified)"
        )
        
        current_feature_idx = feature_options.index(st.session_state.current_feature) if st.session_state.current_feature in feature_options else 0
        st.session_state.current_feature = st.selectbox(
            "Choose an action:", feature_options, index=current_feature_idx, key="sb_feature_select"
        )
        
        if st.session_state.current_feature == "Translate Text":
            st.text_input("Source Language", value="English", key="source_lang_widget")
            st.text_input("Target Language", value="French", key="target_lang_widget")
        
        st.markdown("---")
        if st.button("Clear Editor", key="btn_clear_editor_final"):
            UIComponents._clear_editor_state()
    
    @staticmethod
    def _create_history_tab():
        """Create transformation history tab"""
        st.header("📜 Transformation History")
        if not st.session_state.transformation_history:
            st.caption("No transformations yet.")
        else:
            history_container = st.container(height=400)
            for i, item in enumerate(reversed(st.session_state.transformation_history)):
                with history_container.expander(f"{item['timestamp']}: {item['feature']}"):
                    st.text_area("Original:", value=item['original_text'], height=100, disabled=True, key=f"hist_orig_{i}")
                    st.text_area("Transformed/Analyzed:", value=item['transformed_text'], height=100, disabled=True, key=f"hist_trans_{i}")
                    if st.button("🔁 Revert Editor to this Original", key=f"revert_btn_{i}"):
                        UIComponents._revert_to_history(item)
            
            st.markdown("---")
            if st.button("Clear History", key="btn_clear_history"):
                UIComponents._clear_history()
    
    @staticmethod
    def _create_logs_tab():
        """Create application logs tab"""
        st.header("📊 App Logs")
        log_container = st.container(height=400)
        for log_entry in reversed(st.session_state.app_logs):
            log_container.text(log_entry)
    
    @staticmethod
    def _clear_editor_state():
        """Clear editor and associated state"""
        st.session_state.user_text_area_value = EDITOR_PLACEHOLDER
        st.session_state.file_processing_status = None
        st.session_state.last_processed_file_id = None
        st.session_state.ai_detection_result = None
        st.session_state.plagiarism_check_result = None
        st.session_state.custom_transform_instruction = ""
        st.session_state.audio_input_processed_id = None
        st.session_state.widget_key_suffix += 1
        create_log_entry("Editor cleared, including audio processing state and widget keys.")
        st.rerun()
    
    @staticmethod
    def _revert_to_history(history_item):
        """Revert editor to historical state"""
        st.session_state.user_text_area_value = history_item['original_text']
        st.session_state.ai_detection_result = None
        st.session_state.plagiarism_check_result = None
        create_log_entry(f"Reverted text area to history from {history_item['timestamp']}")
        st.rerun()
    
    @staticmethod
    def _clear_history():
        """Clear transformation history"""
        st.session_state.transformation_history = []
        st.session_state.thread_id = f"thread_{os.urandom(8).hex()}"
        create_log_entry("Transformation history cleared.")
        st.rerun()
    
    @staticmethod
    def display_ai_detection_result():
        """Display AI content detection results"""
        if not st.session_state.ai_detection_result:
            return
        
        res = st.session_state.ai_detection_result
        st.markdown("---")
        st.markdown("#### 🔬 AI Content Analysis")
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.metric(label="AI Likelihood", value=f"{res.get('percentage', 0)}%")
            st.progress(res.get('percentage_float', 0.0))
        with col2:
            st.markdown("**Justification:**")
            st.markdown(res.get('justification', "Not available."))
        st.markdown("---")
    
    @staticmethod
    def display_plagiarism_result():
        """Display plagiarism check results"""
        if not st.session_state.plagiarism_check_result:
            return
        
        res = st.session_state.plagiarism_check_result
        st.markdown("---")
        st.markdown("#### 🔍 Plagiarism Check Result")
        st.metric(label="Overall Similarity", value=str(res.get('overall_similarity', "N/A")))
        if res.get('percentage_float') is not None:
            st.progress(res.get('percentage_float', 0.0))
        
        st.markdown("**Matched Snippets:**")
        matched_snippets_text = res.get('matched_snippets', "None found.")
        if isinstance(matched_snippets_text, list):
            for snippet_info in matched_snippets_text:
                st.markdown(f"- {snippet_info}")
        else:
            st.markdown(matched_snippets_text)
        
        st.markdown("**Justification:**")
        st.markdown(res.get('justification', "Not available."))
        st.markdown("---")

def main():
    """Main application entry point"""
    # Initialize application
    st.set_page_config(page_title="Qwen Quill - AI Writing Assistant", layout="wide", initial_sidebar_state="expanded")
    initialize_session_state()
    configure_page_styling()
    
    # Load configuration
    config = load_environment_variables()
    setup_external_services(config)
    
    # Create workflow
    app_runnable = create_workflow_graph()
    
    # Initialize processors
    file_processor = FileProcessor()
    audio_processor = AudioProcessor(config['assemblyai_key'])
    result_processor = ResultProcessor()
    
    # Create sidebar
    with st.sidebar:
        st.sidebar.title("🛠️ Controls & Info")
        UIComponents.create_sidebar_tabs()
    
    # Main panel header
    try:
        st.image("QwenQuill.png", width=250)
    except Exception as e:
        create_log_entry(f"Could not load logo 'QwenQuill.png'. {e}", level="ERROR")
        st.title("🖋️ Qwen Quill")
    
    st.caption("AI Writing Assistant | Powered by Qwen 3 | OpenRouter and Cerebras Collaboration | VectronixAI")

    # Display file processing status
    if st.session_state.file_processing_status:
        status_type = st.session_state.file_processing_status.get("type", "info")
        message = st.session_state.file_processing_status.get("message", "")
        if status_type == "error":
            st.error(message)
        elif status_type == "success":
            st.success(message)
        else:
            st.info(message)
        st.session_state.file_processing_status = None

    # Text editor
    st.markdown("### 📝 Your Text Editor")
    user_typed_text = st.text_area(
        "Edit your text below. Use features from the sidebar to transform it.",
        value=st.session_state.user_text_area_value,
        key="main_text_area_key_v13",
        height=400,
        label_visibility="collapsed"
    )
    
    if user_typed_text != st.session_state.user_text_area_value:
        st.session_state.user_text_area_value = user_typed_text
        st.session_state.ai_detection_result = None
        st.session_state.plagiarism_check_result = None

    # Custom instruction input for advanced transformations
    if st.session_state.current_feature == "Advanced Transformations":
        st.session_state.custom_transform_instruction = st.text_input(
            "Describe the transformation you want to apply:",
            value=st.session_state.custom_transform_instruction,
            placeholder="e.g., Make this sound more formal, Explain this to a child, Turn into a poem...",
            key="custom_transform_input"
        )

    # Display results
    UIComponents.display_ai_detection_result()
    UIComponents.display_plagiarism_result()

    # Control buttons
    buttons_cols = st.columns([0.33, 0.33, 0.34])
    
    with buttons_cols[0]:
        uploaded_file_widget = st.file_uploader(
            "📎 Attach & Append",
            type=["txt", "pdf", "docx", "png", "jpg", "jpeg"],
            key=f"file_uploader_key_main_v13_{st.session_state.widget_key_suffix}",
            label_visibility="collapsed"
        )
    
    with buttons_cols[1]:
        if not audio_processor.is_available:
            st.markdown(
                "<p class='disabled-voice-button' title='AssemblyAI API Key not configured'>🎤 Voice (Off)</p>",
                unsafe_allow_html=True
            )
        else:
            audio_file_data = st.audio_input(
                "🎤 Record Voice",
                key=f"mic_input_assemblyai_v13_{st.session_state.widget_key_suffix}"
            )
    
    with buttons_cols[2]:
        process_button_clicked = st.button("✨ Apply Feature", key="process_btn_key_main_v13", use_container_width=True)

    # Handle voice input
    if audio_processor.is_available and 'audio_file_data' in locals() and audio_file_data is not None:
        current_audio_input_file_id = audio_file_data.file_id
        if st.session_state.get('audio_input_processed_id') != current_audio_input_file_id:
            create_log_entry(f"New audio input detected (file_id: {current_audio_input_file_id}). Processing with AssemblyAI...")
            spinner_msg = st.empty()
            spinner_msg.info("🎙️ Transcribing audio with AssemblyAI...")
            
            try:
                audio_bytes = audio_file_data.getvalue()
                transcribed_text, error_message = audio_processor.transcribe_audio(audio_bytes, current_audio_input_file_id)
                
                if error_message:
                    st.session_state.file_processing_status = {"type": "error", "message": error_message}
                elif transcribed_text:
                    current_text_val = st.session_state.user_text_area_value
                    if current_text_val == EDITOR_PLACEHOLDER:
                        st.session_state.user_text_area_value = transcribed_text
                    else:
                        st.session_state.user_text_area_value = (current_text_val + " " + transcribed_text).strip()
                    st.session_state.file_processing_status = {"type": "success", "message": "Text from voice appended to editor."}
                    st.session_state.ai_detection_result = None
                    st.session_state.plagiarism_check_result = None
                
                st.session_state.audio_input_processed_id = current_audio_input_file_id
                
            finally:
                spinner_msg.empty()
                st.rerun()

    # Handle file uploads
    if uploaded_file_widget is not None:
        new_file_identifier = f"{uploaded_file_widget.name}_{uploaded_file_widget.size}"
        if st.session_state.get('last_processed_file_id') != new_file_identifier:
            original_filename = uploaded_file_widget.name
            create_log_entry(f"File '{original_filename}' detected. Attempting to extract text.")
            
            current_text_in_area = st.session_state.user_text_area_value
            text_to_replace_or_append = ""
            if current_text_in_area == EDITOR_PLACEHOLDER:
                text_to_replace_or_append = ""
            else:
                text_to_replace_or_append = current_text_in_area + ("\n\n" if current_text_in_area.strip() else "")
            
            try:
                file_bytes = uploaded_file_widget.getvalue()
                content_to_add = ""
                file_type = uploaded_file_widget.type
                error_message = None

                if file_type == "text/plain":
                    content_to_add = file_processor.process_text_file(file_bytes)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    content_to_add = file_processor.process_docx_file(file_bytes)
                elif file_type == "application/pdf":
                    content_to_add = file_processor.process_pdf_file(file_bytes)
                elif file_type in ["image/png", "image/jpeg"]:
                    content_to_add, error_message = file_processor.process_image_file(file_bytes, original_filename)

                if error_message:
                    st.session_state.file_processing_status = {"type": "error", "message": error_message}
                elif content_to_add or file_type not in ["image/png", "image/jpeg"]:
                    st.session_state.user_text_area_value = text_to_replace_or_append + content_to_add
                    create_log_entry(f"Text from '{original_filename}' processed.")
                    if not st.session_state.file_processing_status:
                        st.session_state.file_processing_status = {"type": "success", "message": f"Text from '{original_filename}' integrated into editor."}
                
                st.session_state.last_processed_file_id = new_file_identifier
                st.session_state.ai_detection_result = None
                st.session_state.plagiarism_check_result = None
                st.rerun()
                
            except Exception as e:
                create_log_entry(f"Error processing file '{original_filename}': {e}", level="ERROR")
                st.session_state.file_processing_status = {"type": "error", "message": f"Error processing '{original_filename}': {e}"}
                st.session_state.last_processed_file_id = new_file_identifier
                st.rerun()

    # Handle main processing
    if process_button_clicked:
        original_text_for_history = st.session_state.user_text_area_value
        if not original_text_for_history.strip() or original_text_for_history == EDITOR_PLACEHOLDER:
            st.warning("Text area is empty or contains only placeholder. Please type or upload content.")
        else:
            selected_feature = st.session_state.current_feature
            create_log_entry(f"User initiated '{selected_feature}'. Text length: {len(original_text_for_history)}")
            st.session_state.ai_detection_result = None
            st.session_state.plagiarism_check_result = None

            feature_args_for_graph = {"current_feature_name": selected_feature}
            
            if selected_feature == "Translate Text":
                feature_args_for_graph["source_language"] = st.session_state.get("source_lang_widget", "English")
                feature_args_for_graph["target_language"] = st.session_state.get("target_lang_widget", "French")
            elif selected_feature == "Advanced Transformations":
                custom_instruction = st.session_state.get("custom_transform_instruction", "").strip()
                if not custom_instruction:
                    st.warning("Please describe the advanced transformation you want to apply in the input box below the editor.")
                    st.stop()
                feature_args_for_graph["custom_instruction"] = custom_instruction

            current_graph_input = {
                "input_text_for_llm": original_text_for_history,
                "feature_args": feature_args_for_graph,
                "messages": []
            }
            config_dict = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            try:
                with st.spinner(f"VectronixAI is applying '{selected_feature}'..."):
                    response_state = app_runnable.invoke(current_graph_input, config=config_dict)
                
                ai_response_content = "Sorry, the AI could not process this request."
                if response_state and 'messages' in response_state and response_state['messages']:
                    final_ai_message = response_state['messages'][-1]
                    if isinstance(final_ai_message, AIMessage):
                        ai_response_content = final_ai_message.content
                        
                        if ai_response_content == "[NO_INPUT_PROVIDED_ERROR]":
                            st.warning("No text was provided to the AI.")
                        elif selected_feature == "AI Content Detection":
                            st.session_state.ai_detection_result = result_processor.process_ai_detection_result(ai_response_content)
                            create_log_entry(f"AI Detection result: {st.session_state.ai_detection_result['percentage']}% - {st.session_state.ai_detection_result['justification']}")
                        elif selected_feature == "Plagiarism Check (Simplified)":
                            st.session_state.plagiarism_check_result = result_processor.process_plagiarism_result(ai_response_content)
                            create_log_entry(f"Plagiarism Check: {st.session_state.plagiarism_check_result['overall_similarity']}")
                        else:
                            st.session_state.user_text_area_value = ai_response_content
                    else:
                        ai_response_content = "Received an unexpected response structure from AI."

                # Add to history
                transformed_text_for_history = (
                    st.session_state.ai_detection_result["raw_output"]
                    if selected_feature == "AI Content Detection" and st.session_state.ai_detection_result
                    else st.session_state.plagiarism_check_result["raw_output"]
                    if selected_feature == "Plagiarism Check (Simplified)" and st.session_state.plagiarism_check_result
                    else ai_response_content
                )
                
                st.session_state.transformation_history.append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "feature": selected_feature,
                    "original_text": original_text_for_history,
                    "transformed_text": transformed_text_for_history
                })
                
                if selected_feature not in ["AI Content Detection", "Plagiarism Check (Simplified)"]:
                    create_log_entry(f"Transformation '{selected_feature}' applied. Output (100char): {ai_response_content[:100]}")
                st.rerun()
                
            except Exception as e:
                create_log_entry(f"LangGraph processing error during '{selected_feature}': {e}", level="ERROR")
                st.error(f"An error occurred: {e}. Check logs.")
                st.session_state.transformation_history.append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "feature": selected_feature,
                    "original_text": original_text_for_history,
                    "transformed_text": f"[ERROR DURING TRANSFORMATION: {e}]"
                })
                st.rerun()

if __name__ == "__main__":
    main()
