"""
Chronos Story Director - Backend Engine (Logic Layer)
=====================================================
The "Brain" of the operation.
- Handles all AI Processing (Scene Drafting, Analysis, Chat).
- Manages Workflows (LangGraph).
- Delegates persistence/storage to 'database_manager.py'.
"""

import os
import re
import json
import shutil
import datetime
import glob
from io import BytesIO
from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv

# --- INTERNAL IMPORTS ---
try:
    from . import database_manager as db
except ImportError:
    import database_manager as db

# --- THIRD-PARTY DEPENDENCIES ---
from google import genai as new_genai
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from fpdf import FPDF
from ebooklib import epub

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- TYPE DEFINITIONS ---
class StoryState(TypedDict):
    """
    State schema for the scene generation workflow (LangGraph).
    """
    profile_name: str
    chapter_num: Optional[int]
    part_num: Optional[int]
    year: int
    date_str: str
    time_str: str 
    scene_title: str
    scene_brief: str
    current_draft: str
    revision_count: int
    is_grounded: bool
    recent_context: str 
    banned_words: str
    use_fog_of_war: bool
    context_files: List[str]
    critique_notes: str

# ==========================================
# 1. API PROXY LAYER (Bridge to DB Manager)
# ==========================================
def get_paths(profile): return db.get_paths(profile)
def list_profiles(): return db.list_profiles()
def ensure_profile_structure(name): return db.ensure_profile_structure(name)
def get_story_settings(profile): return db.get_story_settings(profile)
def update_story_setting(p, k, v): return db.update_story_setting(p, k, v)
def get_all_files_list(profile): return db.get_all_files_list(profile)
def read_file_content(p, f): return db.read_file_content(p, f)
def get_world_state(p): return db.get_world_state(p)
def save_world_state(p, s): return db.save_world_state(p, s)
def get_fragments(p, c): return db.get_fragments(p, c)
def add_fragment(p, n, c, t): return db.add_fragment(p, n, c, t)
def update_fragment(p, i, c): return db.update_fragment(p, i, c)
def delete_fragment(p, i): return db.delete_fragment(p, i)
def rename_fragment(p, i, n): return db.rename_fragment(p, i, n)
def get_chat_history(p): return db.get_chat_history(p)
def save_chat_message(p, r, c): return db.save_chat_message(p, r, c)
def clear_chat_history(p): return db.clear_chat_history(p)
def get_recent_faction_memory(p, f, l=3): return db.get_recent_faction_memory(p, f, l)
def get_all_faction_memories(p): return db.get_all_faction_memories(p)
def update_faction_reaction(p, i, t, f): return db.update_faction_reaction(p, i, t, f)
def delete_faction_reaction(p, i): return db.delete_faction_reaction(p, i)
def save_faction_reaction(p, f, t, s): return db.save_faction_reaction(p, f, t, s)
def get_recent_faction_memory(p, f, l=3): return db.get_recent_faction_memory(p, f, l)
def add_project(p, n, d, f): return db.add_project(p, n, d, f)
def update_project(p, i, pr, no, nn=None, nd=None): return db.update_project(p, i, pr, no, nn, nd)
def complete_project(p, i, l, t="Fact"): return db.complete_project(p, i, l, t)

def get_next_chapter_number(profile_name):
    """Calculates the next available chapter number based on existing files."""
    files = db.get_all_files_list(profile_name)
    max_ch = 0
    for f in files:
        match = re.search(r'Ch(?:apter)?_?(\d+)', f, re.IGNORECASE)
        if match:
            try:
                num = int(match.group(1))
                if num > max_ch: max_ch = num
            except: pass
    return max_ch + 1

# ==========================================
# 2. SETTINGS & AI FACTORY MODULE
# ==========================================

# --- CONFIGURATION PROXIES ---
# These allow the API to read/write settings via the Database Manager

def get_story_settings(profile_name: str) -> dict:
    """Retrieves configuration (Time system, Models, etc.) from DB."""
    return db.get_story_settings(profile_name)

def update_story_setting(profile_name: str, key: str, value: str):
    """Updates a specific configuration key in the DB."""
    db.update_story_setting(profile_name, key, value)

# --- AI MODEL MANAGEMENT ---

class MockResponse:
    def __init__(self, text): self.content = text
class MockLLM:
    def invoke(self, *args, **kwargs): return MockResponse("⚠️ SYSTEM ERROR: API Key missing.")

def list_available_models_all() -> List[str]:
    """
    Dynamically lists ONLY the AI models actually available to your API keys.
    Connects to Google, OpenAI, and Anthropic to fetch real-time lists.
    """
    models = []
    
    # 1. Google Gemini Models
    if GOOGLE_API_KEY:
        try:
            client = new_genai.Client(api_key=GOOGLE_API_KEY)
            for m in client.models.list():
                # Safely get supported methods
                methods = getattr(m, 'supported_generation_methods', [])
                model_name = m.name.replace("models/", "")
                
                # Check for content generation support
                if methods and "generateContent" in methods:
                    models.append(model_name)
                elif "gemini" in model_name.lower() and "embedding" not in model_name.lower():
                    models.append(model_name)
                    
        except Exception as e: 
            print(f"Google Model List Error: {e}")
            # Fallback for Google
            models.extend(["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"])
    
    # 2. OpenAI Models
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            for m in client.models.list():
                if "gpt" in m.id.lower() or "o1" in m.id.lower():
                    models.append(m.id)
        except Exception as e:
            print(f"OpenAI Model List Error: {e}")

    # 3. Anthropic Models (Dynamic Fetch)
    if ANTHROPIC_API_KEY:
        try:
            import anthropic 
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            # Fetch list from API
            for m in client.models.list():
                if "claude" in m.id.lower():
                    models.append(m.id)
                    
        except ImportError:
            print("Anthropic library not installed. Run: pip install anthropic")
        except Exception as e:
            print(f"Anthropic Model List Error: {e}")
            # Fallback if connection fails
            models.extend(["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"])

    return sorted(list(set(models)))

_llm_cache = {} 

def get_llm(profile_name: str, task_type: str = "scene", settings: Optional[dict] = None):
    """
    Factory: Returns the correct LLM client based on the Task and Profile Settings.
    """
    if settings is None:
        settings = db.get_story_settings(profile_name)
    
    model_map = {
        "scene": "model_scene",
        "chat": "model_chat",
        "reaction": "model_reaction",
        "analysis": "model_analysis",
        "retrieval": "model_retrieval"
    }
    target_key = model_map.get(task_type, "model_chat")
    model_name = settings.get(target_key, "gemini-2.5-flash")
    
    # --- CHECK CACHE ---
    cache_key = f"{model_name}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    client = None

    # --- CREATE CLIENT ---
    
    # 1. Google Gemini
    if "gemini" in model_name.lower() and GOOGLE_API_KEY:
        client = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY)
    
    # 2. OpenAI GPT
    elif ("gpt" in model_name.lower() or "o1" in model_name.lower()) and OPENAI_API_KEY:
        try:
            client = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)
        except ImportError:
            print("OpenAI library not installed.")

    # 3. Anthropic Claude
    elif "claude" in model_name.lower() and ANTHROPIC_API_KEY and ChatAnthropic:
        client = ChatAnthropic(model=model_name, api_key=ANTHROPIC_API_KEY)

    # 4. Fallback (Default to Gemini Flash if available)
    if not client and GOOGLE_API_KEY:
        client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
        
    # --- SAVE TO CACHE & RETURN ---
    if client:
        _llm_cache[cache_key] = client 
        return client
        
    return MockLLM()

# ==========================================
# 3. SMART RETRIEVAL & HELPERS
# ==========================================

def generate_file_metadata(profile_name: str, content: str) -> str:
    """
    AI Summarizer: Creates a dense, keyword-rich metadata string 
    for the Librarian to use during Smart Retrieval.
    """
    if not content or len(content.strip()) < 50:
        return ""
        
    prompt = f"""
    TASK: Generate searchable metadata for the text below.
    
    INSTRUCTION: Read the text and extract:
    1. A single 1-sentence summary of the main event.
    2. A comma-separated list of all proper nouns (Characters, Locations, Factions, Unique Items).
    
    OUTPUT FORMAT STRICTLY AS:
    Summary: [sentence]
    Entities: [Name1, Name2, Name3]
    
    TEXT:
    {content[:3000]}  # Limit to 3000 chars to save tokens and speed up processing
    """

    llm = get_llm(profile_name, "chat")
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return res
    except Exception as e:
        print(f"Metadata Generation Error: {e}")
        return ""

def get_relevant_fragment_ids(profile_name, user_query, doc_types=None):
    """
    Scans the 'Table of Contents' (Titles + Metadata) and asks the AI 
    which entries are relevant to the user's query.
    """
    rows = db.get_fragments(profile_name, doc_type=None)
    
    if not rows: return []

    # Format the "Menu" for the AI
    toc_list = []
    for r in rows:
        # r[3] is the 'type' column
        if doc_types and r[3] not in doc_types:
            continue
            
        # r[4] is the 'metadata' column. We add it if it exists.
        meta_text = ""
        if len(r) > 4 and r[4]:
            # Replace newlines with spaces so it stays on one line in the TOC
            clean_meta = r[4].replace('\n', ' ')
            meta_text = f" | {clean_meta}"
            
        toc_list.append(f"ID: {r[0]} | Title: {r[1]} ({r[3]}){meta_text}")
    
    if not toc_list: return []
    
    # Limit menu size to save tokens (keep most recent 150 entries)
    toc_str = "\n".join(toc_list[:150])

    prompt = f"""
    ROLE: Database Librarian.
    TASK: Select relevant document IDs based on the user's need.
    
    *** AVAILABLE DOCUMENTS & METADATA ***
    {toc_str}
    
    *** USER SCENARIO / QUERY ***
    "{user_query}"
    
    *** INSTRUCTION ***
    Analyze the scenario. Identify which documents contain necessary background info based on their Title or Metadata.
    - If the user mentions a specific character, location, or event, check the 'Entities' and 'Summary' to find the right file.
    - Select ONLY highly relevant items.
    - Max 7 items.
    
    OUTPUT FORMAT: JSON List of integers ONLY. Example: [1, 14, 22]
    If nothing is relevant, output: []
    """
    
    llm = get_llm(profile_name, "retrieval") 
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content
        ids = _extract_json(res) 
        if isinstance(ids, list):
            return ids
        return []
    except Exception as e:
        print(f"Smart Retrieval Error: {e}")
        return []

def resolve_faction_alias(profile_name, user_input):
    """
    Maps a vague user input (e.g. "The Spies") to a specific Faction Name 
    existing in the database (e.g. "The Guild of Whispers").
    """
    # Use DB Manager to get unique faction names
    existing_factions = db.get_distinct_factions(profile_name)
    
    if not existing_factions: return user_input

    # The "Entity Resolution" Prompt
    prompt = f"""
    TASK: Entity Resolution.
    USER INPUT: "{user_input}"
    KNOWN FACTIONS: {json.dumps(existing_factions)}
    
    INSTRUCTION: Which 'Known Faction' is the user referring to?
    - If it's a nickname (e.g. "The Cops" -> "City Watch"), map it.
    - If it's ambiguous, pick the closest match.
    - If it's a NEW faction not in the list, return "NEW".
    
    OUTPUT: The exact string from the Known Factions list, or "NEW".
    """
    
    llm = get_llm(profile_name, "retrieval")
    res = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    res = res.replace('"', '').replace("'", "")
    
    if res == "NEW" or res not in existing_factions:
        return user_input
    return res

def get_last_scenes(profile_name):
    """Retrieves the trailing context (last 3 scenes) for continuity."""
    files = db.get_all_files_list(profile_name)
    
    if not files: return "NO SCENES."
    
    # Files are sorted Newest -> Oldest. We want the chronological order (Old -> New) of the 3 most recent.
    recent_files = files[:3][::-1] 
    
    context = ""
    for f in recent_files:
        content = db.read_file_content(profile_name, f)
        context += f"\n=== PREV: {f} ===\n{content[:3000]}\n"
    return context

def get_global_context(profile_name: str):
    """Retrieves immutable context layers (Rules, Plan, Spoilers)."""
    # Fetch by type using DB Manager
    r_rows = db.get_fragments(profile_name, "Rulebook")
    rules = "\n\n".join([r[2] for r in r_rows])
    
    p_rows = db.get_fragments(profile_name, "Plan")
    plan = p_rows[0][2] if p_rows else "NO PLAN ESTABLISHED."
    
    s_rows = db.get_fragments(profile_name, "Spoiler")
    spoilers = [r[2] for r in s_rows]
    
    return rules, plan, spoilers

# ==========================================
# 4. SCENE CREATOR ENGINE
# ==========================================

# --- HELPERS ---

def extract_dynamic_spoilers(plan: str, year: int, profile_name: str, settings: Optional[dict] = None) -> List[str]:
    """
    Parses future events from the 'Plan' to prevent context leakage into the current narrative.
    """
    if not plan or plan == "NO PLAN ESTABLISHED.":
        return []
        
    prompt = f"List FUTURE events after {year} from: {plan}. OUTPUT: Comma-separated."
    llm = get_llm(profile_name, "chat", settings=settings)
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        return [x.strip() for x in response.split(',')]
    except Exception:
        return []

def infer_header_data(brief: str, prev_context: str, settings: dict, profile_name: str) -> dict:
    """
    Estimates the narrative date/time for the scene based on recent context using an LLM.
    """
    prompt = f"""
    TASK: Calculate Date/Time/Year.
    BRIEF: {brief}
    CONTEXT END: {prev_context[-500:]}
    DEFAULT TIMEZONE: {settings.get('default_timezone', '')}
    OUTPUT JSON ONLY: {{ "year": 1984, "date": "March 6", "time": "14:00 CST" }}
    """
    llm = get_llm(profile_name, "chat", settings=settings)
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content
        return _extract_json(res)
    except Exception:
        return {}

def auto_generate_title(profile_name: str, draft_text: str, brief: str) -> str:
    """
    Generates a short, evocative title based on the generated scene content.
    """
    prompt = f"""
    TASK: Create a Title.
    SCENE BRIEF: {brief}
    SCENE CONTENT START: {draft_text[:1000]}...
    
    INSTRUCTION: Generate a short, punchy, dramatic title (max 6 words) for this scene. 
    Examples: "The Red Wedding", "Midnight at the Docks", "Protocol Omega".
    OUTPUT: The title text ONLY. No quotes.
    """
    llm = get_llm(profile_name, "chat")
    try:
        return llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        return "Untitled Scene"

# --- CORE GENERATION LOGIC ---

def draft_scene(state: StoryState) -> dict:
    """
    Workflow Node 1: Narrative Drafting (Adaptive Realism).
    
    Constructs a comprehensive prompt including World State, Rules, 
    Global Context, and Narrative Continuity to generate the next scene.
    """
    profile = state['profile_name']
    brief = state['scene_brief']
    chapter = state.get('chapter_num')
    part = state.get('part_num', 1)  # Default to Part 1 if unspecified

    settings = db.get_story_settings(profile)
    state_tracking = db.get_world_state(profile)
    
    # 1. Retrieve Global Context (Rules, Plan, Spoilers)
    rules, plan, db_spoilers = get_global_context(profile)
    
    # 2. Smart Retrieval (RAG)
    print(f"  [Librarian] Scanning Knowledge Base for: '{brief[:50]}...'")
    relevant_ids = get_relevant_fragment_ids(
        profile, 
        user_query=brief, 
        doc_types=["Lore", "Fact", "Rulebook", "Scene"]
    )
    
    smart_context_str = db.get_content_by_ids(profile, relevant_ids)
    if not smart_context_str:
        smart_context_str = "No specific historical records found for this scene."

    # 3. Header & Continuity Logic
    #    If this is Part 2+, fetch the text of previous parts to ensure flow.
    chapter_line = ""
    partition_context = ""

    if chapter is not None:
        chapter_line = f"CHAPTER: {chapter}"
        if part and int(part) > 1:
            chapter_line += f" (PART {part})"
            
            print(f"  [Engine] Fetching continuity for Chapter {chapter}, Part {part}...")
            paths = db.get_paths(profile)
            # Locate previous parts (e.g., Ch01_Part_1...)
            for p in range(1, int(part)):
                pattern = os.path.join(paths['output'], f"Ch{int(chapter):02d}_Part_{p}_*.txt")
                found_files = glob.glob(pattern)
                if found_files:
                    with open(found_files[0], 'r', encoding='utf-8') as f:
                        partition_context += f"\n--- CHAPTER {chapter} PART {p} (PREVIOUS) ---\n{f.read()}\n"

    # 4. Dynamic Spoiler Injection
    #    Prevents the AI from referencing future events defined in the Plan.
    dynamic_spoilers = extract_dynamic_spoilers(plan, state['year'], profile, settings=settings) 
    all_banned = list(set(db_spoilers + dynamic_spoilers))
    
    # 5. Chronology & Era Detection
    use_time_system = settings.get('use_time_system', 'true').lower() == 'true'
    header = ""
    era_display = "Undefined (Infer Tech Level from Lore)"
    
    if use_time_system and state['year'] > 0:
        header = f"{state['date_str']}, {state['year']}"
        era_display = f"{state['year']}"
        if state['time_str']: header += f"\n{state['time_str']}"

    # 6. Timeline Logic (Multiverse Support)
    timeline_section = ""
    if settings.get('use_timelines', 'true').lower() == 'true':
        timelines_list = state_tracking.get("Timelines", [])
        if timelines_list:
            timeline_section = "ACTIVE TIMELINES (MULTIVERSE):\n"
            for t in timelines_list:
                timeline_section += f"- {t.get('Name', 'Unknown')}: {t.get('Description', '')}\n"
    
    # 7. World Variables (Physics/Mechanics)
    variables_section = ""
    world_vars = state_tracking.get("World Variables", [])
    if world_vars:
        variables_section = "*** WORLD MECHANICS (STRICT) ***\n"
        for v in world_vars:
            variables_section += f"- {v.get('Name', 'Var')}: {v.get('Value', '0')} (RULE: {v.get('Mechanic', '')})\n"

    # 8. Privacy Protocol (Fog of War)
    privacy_protocol = ""
    if state.get('use_fog_of_war', False):
        privacy_protocol = """
        *** PRIVACY & FOG OF WAR PROTOCOL ***
        RULE: Wrap private interactions (whispers, internal thoughts, secure rooms) in [[PRIVATE]] ... [[/PRIVATE]] tags.
        EXAMPLE:
        They stood in the public square. "Everything is fine," he announced loudly.
        [[PRIVATE]]
        Once inside the secure room, he slumped against the door. "We are doomed," he whispered.
        [[/PRIVATE]]
        """

    # 9. Construct Final Prompt (Adaptive Realism)
    #    Note: partition_context is injected to ensure continuity between parts.
    prompt = f"""
    ROLE: Novelist (Third Person Limited).
    CHARACTER: {settings.get('protagonist', 'Protagonist')}.
    {chapter_line}  <-- ONLY INJECT IF EXISTS
    CURRENT CALENDAR YEAR: {era_display}
    
    *** NARRATIVE CONTINUITY (PREVIOUS PARTS) ***
    {partition_context}
    
    *** NARRATIVE LOGIC & TECH-LEVEL (HIERARCHY OF TRUTH) ***
    1. LORE PRIORITY (ABSOLUTE):
       - The 'Story Bible' and 'Rules' are the primary source.
       - If Lore says "Year 407" features Flying Airships, then Airships exist.
       - Do NOT assume "Year 407" means "Real World 407 AD" unless the Lore explicitly confirms it is Earth.
       
    2. DETERMINING THE TECH LEVEL:
       - CHECK LORE FIRST: Scan the Lore below. Does it mention magic, advanced tech, or specific tools? USE THAT.
       - REAL WORLD FALLBACK (Conditional): ONLY if the Lore is SILENT and the setting appears to be Earth, use real-world history for the year {era_display}.
         * Example: Year 1990 + "New York" -> Use Real 1990 Tech (VHS, Landlines).
         * Example: Year 407 + "Kingdom of Asura" -> IGNORE Real 407 AD. Use the Fantasy Logic defined in Rules.
       
    3. REALISM WITHIN CONTEXT: 
       - Once the Tech Level is set (Fantasy or Real), maintain internal consistency.
       - If it's Fantasy, describe the fantasy elements realistically (e.g. the hum of the magic crystal).

    *** WORLD LAWS & MECHANICS (STRICT) ***
    {rules}
    {privacy_protocol}
    {variables_section}
    
    *** STRATEGIC PLAN ***
    {plan}

    *** RELEVANT LORE & CONTEXT (SMART RETRIEVAL) ***
    {smart_context_str}

    *** WORLD STATE & PROJECTS ***
    {json.dumps(state_tracking)}
    
    *** FORMATTING ***
    {header}
    (Start prose below header. NO Title in body).
    
    {timeline_section}
    
    *** BANNED CONCEPTS (SPOILERS) ***
    [{", ".join(all_banned)}]
    
    *** NARRATIVE CONTEXT (RECENT) ***
    {state['recent_context']}
    
    === MISSION ===
    BRIEF: {state['scene_brief']}
    
    CRITIQUE: {state['critique_notes']}
    """
    
    # 10. Execute Generation
    llm = get_llm(profile, "scene", settings=settings)
    response = llm.invoke([HumanMessage(content=prompt)]).content
    
    return {
        "current_draft": response, 
        "revision_count": state['revision_count'] + 1, 
        "banned_words": ", ".join(all_banned)
    }

def critique_scene(state: StoryState) -> dict:
    """
    Workflow Node 2: Validation.
    Checks adherence to constraints (e.g. banned words) and logic integrity.
    """
    prompt = f"ROLE: Editor. CHECK: Banned [{state['banned_words']}]? DRAFT: {state['current_draft']} OUTPUT: PASS/FAIL"
    llm = get_llm(state['profile_name'], "chat")
    res = llm.invoke([HumanMessage(content=prompt)]).content
    
    if "PASS" in res:
        return {"is_grounded": True, "critique_notes": ""}
    return {"is_grounded": False, "critique_notes": "Found banned content or logic break."}

def generate_scene(
    profile: str, 
    chapter_num: Optional[int], 
    year: int, 
    date_str: str, 
    time_str: str, 
    title: str, 
    brief: str, 
    context_files: List[str], 
    use_fog_of_war: bool,
    part: int = 1
) -> tuple[str, str]:
    """
    Entry point for the scene generation pipeline.
    Initializes the state graph, aggregates context, and executes the workflow.
    """
    # 1. Graph Setup
    workflow = StateGraph(StoryState)
    workflow.add_node("drafter", draft_scene)
    workflow.add_node("validator", critique_scene)
    workflow.set_entry_point("drafter")
    workflow.add_edge("drafter", "validator")
    workflow.add_conditional_edges("validator", lambda s: END if s['is_grounded'] or s['revision_count'] > 2 else "drafter")
    app = workflow.compile()
    
    # 2. Context Assembly
    context_str = ""
    if context_files:
        for fname in context_files:
            if fname == "Auto (Last 3 Scenes)": 
                context_str += get_last_scenes(profile)
            else: 
                context_str += f"\n=== CONTEXT: {fname} ===\n{db.read_file_content(profile, fname)[:5000]}\n"
    else: 
        frags = db.get_fragments(profile, "Lore")
        context_str = f"=== BACKGROUND LORE ===\n{frags[0][2]}" if frags else "NO LORE ESTABLISHED."

    settings = db.get_story_settings(profile)
    
    # 3. Heuristic Time Inference
    use_time = settings.get('use_time_system', 'true').lower() == 'true'
    final_year = year
    final_date = date_str
    final_time = time_str
    
    if use_time and (not final_year or not final_date or not final_time):
        inferred = infer_header_data(brief, context_str, settings, profile)
        if not final_year: final_year = inferred.get('year', 1984)
        if not final_date: final_date = inferred.get('date', "Unknown Date")
        if not final_time: final_time = inferred.get('time', "")

    try: final_year = int(final_year)
    except: final_year = 0

    # 4. Chapter Handling
    enable_chapters = str(settings.get('enable_chapters', 'true')).lower() == 'true'
    if not enable_chapters:
        chapter_num = None
    elif chapter_num is None:
        chapter_num = get_next_chapter_number(profile)

    # INITIAL STATE
    temp_title = title if title else "Untitled Processing..."

    initial_input = {
        "profile_name": profile,
        "chapter_num": chapter_num,
        "part_num": part,
        "year": final_year,
        "date_str": final_date,
        "time_str": final_time,
        "scene_title": temp_title,
        "scene_brief": brief,
        "recent_context": context_str,
        "revision_count": 0,
        "critique_notes": "",
        "is_grounded": False,
        "current_draft": "",
        "banned_words": "",
        "use_fog_of_war": use_fog_of_war,
    }
    
    final_state = app.invoke(initial_input)
    
    # 6. Auto-Title & Persistence
    final_title = title
    if not final_title:
        final_title = auto_generate_title(profile, final_state['current_draft'], brief)

    safe_title = re.sub(r'[\\/*?:"<>|]', "", final_title).replace(" ", "_")

    # Filename Generation (incorporating Part)
    prefix = ""
    if chapter_num is not None:
        part_suffix = f"_Part_{part}" if part and int(part) > 1 else ""
        prefix = f"Ch{int(chapter_num):02d}{part_suffix}_"

    if use_time:
        safe_date = str(final_date).replace(" ", "-")
        filename = f"{prefix}{final_year}-{safe_date}_{safe_title}.txt"
    else:
        filename = f"{prefix}{safe_title}.txt"
    
    paths = db.get_paths(profile)
    filepath = os.path.join(paths['output'], filename)
    
    # Collision Avoidance
    counter = 1
    while os.path.exists(filepath):
        base_name = filename.replace(".txt", "")
        filename = f"{base_name}_{counter}.txt"
        filepath = os.path.join(paths['output'], filename)
        counter += 1
        
    # [ACTION: WRITE TO FILE]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_state['current_draft'])
        
    # [SYNC TO DATABASE WITH METADATA]
    print(f"  [Engine] Generating metadata for {filename}...")
    metadata = generate_file_metadata(profile, final_state['current_draft'])
    db.upsert_scene(profile, filename, final_state['current_draft'], metadata)

    return final_state['current_draft'], filepath

def save_edited_scene(profile: str, filename: str, content: str) -> tuple[bool, str]:
    """
    Overwrites a scene file with manual edits and updates the database with new metadata.
    """
    try:
        paths = db.get_paths(profile)
        filepath = os.path.join(paths['output'], filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        # [Re-generate metadata because the user might have changed important facts]
        print(f"  [Engine] Updating metadata for edited scene {filename}...")
        metadata = generate_file_metadata(profile, content)
        db.upsert_scene(profile, filename, content, metadata)
        
        return True, "Saved successfully."
    except Exception as e:
        return False, str(e)

# --- FILE OPERATIONS (MERGE & DELETE) ---

def merge_specific_files(profile: str, filenames: List[str]) -> str:
    """
    Stitches a user-selected list of files together.
    Archives the source files after a successful merge and updates the DB.
    """
    paths = db.get_paths(profile)
    base_path = paths['output']
    combined_content = ""
    
    # 1. Stitch Content with separator
    for fname in filenames:
        fpath = os.path.join(base_path, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                combined_content += f.read() + "\n\n# # #\n\n"
    
    # 2. Generate New Name
    # Logic: Remove "_Part_X" from the first filename to create the merged title.
    first_name = filenames[0]
    new_name = re.sub(r"_Part_\d+", "", first_name)
    
    if new_name == first_name:
        new_name = "Merged_" + first_name
        
    filepath = os.path.join(base_path, new_name)
    
    # 3. Write new file and Sync to DB
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(combined_content)
        
    # [Sync merged file to DB with fresh metadata]
    print(f"  [Engine] Generating metadata for merged scene {new_name}...")
    metadata = generate_file_metadata(profile, combined_content)
    db.upsert_scene(profile, new_name, combined_content, metadata)
    
    # 4. Archive the original parts
    archive_dir = os.path.join(base_path, "Archive")
    os.makedirs(archive_dir, exist_ok=True)
    
    for fname in filenames:
        source = os.path.join(base_path, fname)
        if os.path.exists(source):
            # Move physical file
            shutil.move(source, os.path.join(archive_dir, fname))
            # [Archive the old scene in the Database]
            db.archive_scene_db(profile, fname)
            
    return new_name


def bulk_delete_files(profile: str, filenames: List[str]) -> int:
    """
    Deletes multiple files in one operation from both disk and Database.
    """
    paths = db.get_paths(profile)
    count = 0
    for fname in filenames:
        fpath = os.path.join(paths['output'], fname)
        if os.path.exists(fpath):
            # 1. Delete physical file
            os.remove(fpath)
            # 2. [Delete from Database]
            db.delete_scene_db(profile, fname)
            count += 1
            
    return count

# ==========================================
# 5. CO-AUTHOR CHAT MODULE
# ==========================================

def get_chat_history(profile_name):
    """Loads chat history from the database via the manager."""
    return db.get_chat_history(profile_name)

def save_chat_message(profile_name, role, content):
    """Appends a message to the persistent chat log."""
    db.save_chat_message(profile_name, role, content)

def clear_chat_history(profile_name):
    """Purges the chat history log."""
    db.clear_chat_history(profile_name)

def run_chat_query(profile_name, user_input):
    """
    Interacts with the Co-Author persona (Adaptive Logic).
    """
    # 1. Retrieve Global Context (Using DB helpers)
    rules, plan, _ = get_global_context(profile_name)
    state = db.get_world_state(profile_name)
    settings = db.get_story_settings(profile_name)

    # 2. Get Recent Narrative
    recent_scenes = get_last_scenes(profile_name)
    
    # 3. Smart Retrieval (Internal Function)
    print(f"  [Co-Author] Researching: '{user_input[:50]}...'")
    relevant_ids = get_relevant_fragment_ids(
        profile_name, 
        user_query=user_input, 
        doc_types=["Lore", "Fact", "Rulebook", "Scene"]
    )
    
    # Efficient Batch Fetch via DB
    smart_knowledge = db.get_content_by_ids(profile_name, relevant_ids)
    if not smart_knowledge:
        smart_knowledge = "No specific database records found (Relying on General Knowledge)."

    # 4. Era / Tech-Level Detection (Standardized)
    use_time_system = settings.get('use_time_system', 'true').lower() == 'true'
    era_display = "Undefined (Infer Tech Level from Lore)"
    if use_time_system and state.get('year', 0) > 0:
        era_display = f"{state['year']}"
    
    # 5. Construct Prompt (HIERARCHY OF TRUTH + CALENDAR AGNOSTIC)
    prompt = f"""
    ROLE: Co-Author & Omniscient Editor.
    
    *** TEMPORAL STATUS ***
    CURRENT CALENDAR YEAR: {era_display}
    
    *** PRIMARY SOURCE OF TRUTH (STORY BIBLE) ***
    {smart_knowledge}
    
    *** WORLD RULES (IMMUTABLE) ***
    {rules}
    
    *** FUTURE PLANS (DRAFTS) ***
    {plan[:3000]}
    
    *** CURRENT WORLD STATE ***
    {json.dumps(state)}
    
    *** RECENT NARRATIVE ***
    {recent_scenes}
    
    *** USER QUERY ***
    "{user_input}"
    
    *** INSTRUCTION & LOGIC ***
    You are an Omniscient Editor. You know real-world history AND the story's lore.
    Follow this HIERARCHY OF TRUTH to answer the query:
    
    1. RANK 1: LORE & RULES (ABSOLUTE TRUTH)
       - If the Story Bible mentions a technology or concept, ACCEPT IT as fact, regardless of the year.
       - If Lore says "Year 407" has airships, then airships exist. Do NOT assume "Year 407" means "Real Earth 407 AD".
    
    2. RANK 2: REAL WORLD KNOWLEDGE (CONDITIONAL FALLBACK)
       - If the Lore is SILENT, check the setting type:
         * IF EARTH-BASED: Use real-world history/science for the year {era_display}. (e.g. 1920 = Prohibition Era).
         * IF FANTASY/ALIEN: Do NOT use Earth history. Infer the logic from the Rules (e.g. "If magic exists, use magic for medicine, not leeches").
    
    3. RANK 3: CHRONOLOGY CHECK (THE SAFETY NET)
       - If the user asks for something that contradicts Rank 1 or Rank 2 (e.g. "iPhone in 1920"), FLAG IT as an anachronism.
       - However, if the user asks for a DEFINITION (e.g. "What is an iPhone?"), answer accurately but note it doesn't exist yet in the story.
    """
    
    llm = get_llm(profile_name, "chat")
    return llm.invoke([HumanMessage(content=prompt)]).content

# ==========================================
# 6. WAR ROOM MODULE
# ==========================================

def run_war_room_simulation(profile, action_input):
    """
    Executes a Monte Carlo strategic simulation (Smart Retrieval).
    
    Upgrade: Now uses the 'Librarian' to find specific historical precedents, 
    enemy capabilities (Lore), and reads recent scenes to understand the 
    immediate tactical situation.
    """
    # 1. Retrieve Global Rules & Plan (Uses local helper wrapping DB)
    rules, plan, _ = get_global_context(profile)
    
    # 2. Retrieve World State (Direct DB call)
    state = db.get_world_state(profile)
    
    # 3. Get Immediate Tactical Context (Uses local helper)
    recent_history = get_last_scenes(profile)
    
    # 4. Smart Retrieval (Strategic Intelligence)
    #    Uses the local function 'get_relevant_fragment_ids' defined earlier in backend.py
    print(f"  [War Room] Gathering Intelligence for: '{action_input[:50]}...'")
    relevant_ids = get_relevant_fragment_ids(
        profile, 
        user_query=f"Strategic analysis of: {action_input}", 
        doc_types=["Lore", "Fact", "Rulebook", "Scene"]
    )
    
    #    Uses DB Manager for efficient batch content fetching
    smart_intel = db.get_content_by_ids(profile, relevant_ids)
    if not smart_intel:
        smart_intel = "No specific intelligence dossiers found."

    # 5. Construct the Dossier
    #    Note: Uses json.dumps for clean data formatting within the prompt
    intel_packet = f"""
    *** CURRENT ASSETS & STATUS ***
    Protagonist Status: {json.dumps(state.get('Protagonist Status', {}))}
    Known Cast & Factions: {json.dumps(state.get('Cast', []))} 
    
    Available Assets: {json.dumps(state.get('Assets', []))}
    Current Skills: {json.dumps(state.get('Skills', []))}
    
    *** IMMEDIATE CONTEXT (RECENT EVENTS) ***
    {recent_history[-4000:]} 
    
    *** RELEVANT KNOWLEDGE (LORE/FACTS) ***
    {smart_intel}
    """
    
    # 6. The "Causality Report" Prompt (Exact Restoration)
    prompt = f"""
    ROLE: Strategic Simulation Engine.
    
    *** WORLD RULES & PHYSICS ***
    {rules}
    
    *** CONTEXT PACKET ***
    {intel_packet}
    
    *** CURRENT GOAL ***
    {plan[:2000]}
    
    *** PROPOSED ACTION ***
    "{action_input}"
    
    *** MISSION ***
    Simulate the consequences of this action based on the World Rules.
    Do not write a story. Write a CAUSALITY REPORT.
    
    *** REPORT FORMAT ***
    ## 📊 Simulation Results
    **Probability of Success:** [0-100%]
    
    ### 1. Direct Consequences (Immediate Outcome)
    * [What happens if the action succeeds/fails?]
    * [Cost (Resources, Health, Reputation, or Time)]
    
    ### 2. Second-Order Effects (The Ripple)
    * [Unintended side effects on Relationships/Factions/Environment]
    * [Systemic shifts (Social, Political, Economic, or Magical)] <-- BROADER SCOPE
    
    ### 3. Critical Risks (Blowback)
    * [Who/What reacts negatively?]
    * [Potential catastrophe?]
    
    ### 4. Verdict
    [Go / No-Go recommendation]
    """
    
    # 7. Execution
    llm = get_llm(profile, "analysis")
    return llm.invoke([HumanMessage(content=prompt)]).content

# ==========================================
# 7. RAG & KNOWLEDGE BASE MODULE
# ==========================================

def get_content_by_ids(profile_name, id_list):
    """
    Retrieves full text content for a specific list of fragment IDs.
    Proxies to the optimized batch fetcher in the database manager.
    """
    return db.get_content_by_ids(profile_name, id_list)

def get_global_context(profile_name: str):
    """
    Retrieves the 'Immutable' context layers that must be present in every generation cycle.
    1. Rules: The physics/magic/laws of the world.
    2. Plan: The strategic direction of the story.
    3. Spoilers: Critical secrets to protect.
    """
    # Fetch rows from DB Manager: (id, filename, content, type)
    # Index 2 is 'content'
    
    # World Rules
    r_rows = db.get_fragments(profile_name, "Rulebook")
    rules = "\n\n".join([r[2] for r in r_rows])
    
    # Strategic Plan
    p_rows = db.get_fragments(profile_name, "Plan")
    plan = p_rows[0][2] if p_rows else "NO PLAN ESTABLISHED."
    
    # Spoilers
    s_rows = db.get_fragments(profile_name, "Spoiler")
    spoilers = [r[2] for r in s_rows]
    
    return rules, plan, spoilers

def get_full_context_data(profile_name: str):
    """
    Retrieves ALL context layers (Lore, Rules, Plans, Facts, Spoilers).
    Used for heavy-duty analysis or deep simulation prompts.
    """
    # Helper to extract content string from rows
    def extract_text(rows): return "\n\n".join([r[2] for r in rows])

    lore = extract_text(db.get_fragments(profile_name, "Lore"))
    rules = extract_text(db.get_fragments(profile_name, "Rulebook"))
    
    # Plan (Limit 1 usually, but here we take all if multiple exist)
    p_rows = db.get_fragments(profile_name, "Plan")
    plan = p_rows[0][2] if p_rows else "NO PLAN."
    
    # Facts (Bulleted list style)
    f_rows = db.get_fragments(profile_name, "Fact")
    facts = "\n".join([f"- {r[2]}" for r in f_rows])
    
    # Spoilers (List)
    s_rows = db.get_fragments(profile_name, "Spoiler")
    spoilers = [r[2] for r in s_rows]
    
    return lore, rules, plan, facts, spoilers

def get_initial_lore(profile_name: str) -> str:
    """Fallback context provider for the initial session if no scenes exist."""
    frags = db.get_fragments(profile_name, "Lore")
    if frags:
        # Return the content of the most recent Lore entry
        return f"=== BACKGROUND LORE ===\n{frags[0][2]}"
    return "NO LORE ESTABLISHED. STARTING FRESH."

# --- CRUD PROXIES (Bridge to Database Manager) ---

def get_fragments(profile_name: str, doc_type: Optional[str] = None):
    """Queries memory fragments."""
    return db.get_fragments(profile_name, doc_type)

def add_fragment(profile_name, filename, content, doc_type):
    """Persists a new document to DB and File System."""
    db.add_fragment(profile_name, filename, content, doc_type)

def update_fragment(profile_name, frag_id, new_content):
    """Updates content in DB and rewrites the file."""
    db.update_fragment(profile_name, frag_id, new_content)

def rename_fragment(profile_name, frag_id, new_filename):
    """Updates the display label and renames the physical file."""
    db.rename_fragment(profile_name, frag_id, new_filename)

def delete_fragment(profile_name, frag_id):
    """Removes from DB and deletes the physical file."""
    db.delete_fragment(profile_name, frag_id)

# ==========================================
# 8. WORLD STATE TRACKER MODULE
# ==========================================

def analyze_state_changes(profile_name, scene_content):
    """
    Executes an LLM analysis of the scene to auto-update the world state (JSON).
    Detects changes in allies, assets, skills, reputation, AND abstract World Variables.
    """
    # Fetch state via DB Manager
    state = db.get_world_state(profile_name)
    
    prompt = f"""
    ROLE: World State Database Manager.
    
    TASK: Analyze the TEXT CONTENT (Scene, Lore, or Plan) and update the JSON STATE.
    
    *** CURRENT STATE ***
    {json.dumps(state)}
    
    *** NARRATIVE SCENE ***
    {scene_content}
    
    *** UPDATE INSTRUCTIONS ***
    1. TIME & DATES (CRITICAL EXECUTION):
       - Compare any detected years against the 'Current_Year' ({state.get('Current_Year', 'Unknown')}).
       - RULE A: "NARRATIVE ONLY": ONLY update 'Current_Year' if the NARRATIVE VOICE confirms the story has actually reached that time.
         * YES: "The year 2030 finally arrived..." (Update to 2030)
         * YES: "Ten years passed..." (Add 10 years)
         * NO: Dialogue references (e.g. "I will be done in 2038") -> IGNORE.
         * NO: Future Plans/Visions (e.g. "He foresaw the crash of 2029") -> IGNORE.
       - RULE B: "FORWARD ONLY": Never update to a year older than Current_Year (Flashback protection).
       - RULE C: BIRTH YEAR: If explicitly mentioned as a fact (e.g. "Born in 1984"), update 'Protagonist Status' -> 'Birth_Year'.

    2. WORLD VARIABLES (CRITICAL):
       - Review the 'World Variables' list in the State.
       - Based on the scene's events, strictly Apply the "Mechanic/Rule" defined for each variable.
       - Example: If 'Federal Heat' rule says "violence increases this", and scene has violence, increase the Value.
       - Output the UPDATED list of variables.

    3. CAST & ROSTER: 
       - Update the 'Cast' list. 
       - For existing characters (match by Name), update 'Role' or 'Tags' if their status changes.
       - If a new MAJOR character appears, add them to 'Cast' (Role='Support').
       - Update 'Loyalty' numbers based on interactions.

    4. ASSETS: Add new resources/locations gained. Mark lost assets as "Destroyed".

    5. SKILLS: Add new skills learned.

    6. ALIASES & REPUTATION:
       - Look for new titles, nicknames, or reputations bestowed upon the protagonist by the public or other characters.
       - Example: If they conquer a city, add "Conqueror of [City]".
       - Example: If they fix the economy, add "The Architect".
       - MERGE these into the existing 'Aliases' string in 'Protagonist Status' (comma-separated).

    7. PROJECTS & RESEARCH:
       - Review the 'Projects' list.
       - If the narrative describes significant work, breakthroughs, or testing related to a project, INCREASE its "Progress" (0-100).
       - Small effort: +5-10%. Major breakthrough: +20-50%. Completion: Set to 100%.
       - If the project is ruined/destroyed, reduce progress.
    
    CRITICAL OUTPUT RULE: 
    You must return the COMPLETE JSON STATE object, including all unchanged fields (Protagonist, Lore, etc.). 
    DO NOT return a partial update. The output must be the full, valid JSON structure.
    """
    
    llm = get_llm(profile_name, "analysis") 
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content
        new_state = _extract_json(res)
        
        # Robust merge logic: If LLM returns partial, merge it. If full, swap it.
        if new_state:
            # Simple check: if key sections are missing, treat as partial update
            if "Protagonist Status" not in new_state:
                state.update(new_state)
                return state
            else:
                return new_state
        return state

    except Exception as e: 
        print(f"Analysis Error: {e}")
        return state

# ==========================================
# 9. NETWORK MAP
# ==========================================

def generate_network_graph(profile: str):
    """
    Constructs the node/edge graph from the new 'Cast' Roster.
    Supports Multi-POV (Constellation) layouts.
    """
    state = db.get_world_state(profile)
    cast = state.get("Cast", [])
    assets = state.get("Assets", [])
    
    nodes = []
    edges = []
    
    # 1. Build Character Nodes
    for char in cast:
        is_pov = char.get("Role") == "POV"
        
        # Determine Visual Category for styling
        category = "Ally"
        if is_pov: category = "Protagonist" 
        elif char.get("Role") == "Antagonist": category = "Enemy"
        
        nodes.append({
            "id": char["id"],
            "type": "customNode",
            "data": { 
                "label": char["Name"],
                "icon": char.get("Icon", "Neutral"),
                "category": category,
                "role": char.get("Role", "Support"),
                "orbit": char.get("Orbit", None)
            },
            # Default pos (Frontend will auto-arrange)
            "position": char.get("ui_pos", {"x": 0, "y": 0}) 
        })

        # 2. Build Character Edges (Links)
        for link in char.get("Links", []):
            target_id = link["targetId"]
            
            # Create a unique sorted ID for the edge so A->B and B->A don't create two lines
            edge_id = f"e-{sorted([char['id'], target_id])[0]}-{sorted([char['id'], target_id])[1]}"
            
            # Only add if not already in edges list
            if not any(e['id'] == edge_id for e in edges):
                edges.append({
                    "id": edge_id,
                    "source": char["id"],
                    "target": target_id,
                    "label": link["type"]
                })

    # 3. Build Asset Nodes
    # Assets orbit the Main POV (or the first found POV) by default
    first_pov = next((c for c in cast if c.get("Role") == "POV"), None)
    
    for i, asset in enumerate(assets):
        asset_id = f"asset_{i}"
        nodes.append({
            "id": asset_id,
            "type": "customNode",
            "data": {
                "label": asset.get("Asset", "Item"),
                "icon": asset.get("Icon", "Resource"),
                "category": "Asset"
            },
            "position": asset.get("ui_pos", {"x": 0, "y": 0})
        })
        
        # Create 'Owns' link
        if first_pov:
            edges.append({
                "id": f"e-{first_pov['id']}-{asset_id}",
                "source": first_pov['id'],
                "target": asset_id,
                "label": "Owns"
            })

    return {"nodes": nodes, "edges": edges}

# ==========================================
# 10. REACTION TOOL & FACTION LOGIC
# ==========================================

def save_faction_reaction(profile_name, faction, text, scene_name):
    """Logs a raw reaction to the database via DB Manager."""
    db.save_faction_reaction(profile_name, faction, text, scene_name)

def get_recent_faction_memory(profile_name, faction, limit=3):
    """Retrieves the last few raw reactions via DB Manager."""
    return db.get_recent_faction_memory(profile_name, faction, limit)

def undo_last_reaction_text(profile_name, filename, faction):
    """
    Removes the last appended reaction for a specific faction from the text file.
    Safeguard: Only deletes if the file explicitly ends with a reaction block for this faction.
    """
    paths = db.get_paths(profile_name)
    filepath = os.path.join(paths['output'], filename)
    
    if not os.path.exists(filepath): return False, "File not found."
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    header_marker = f">>> REACTION: {faction}"
    
    if header_marker not in content:
        return False, "No reaction text found in file."

    parts = content.rsplit(header_marker, 1)
    
    if len(parts) < 2:
        return False, "Could not isolate reaction block."

    clean_content = parts[0].rstrip()

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(clean_content)
        
    db.upsert_scene(profile_name, filename, clean_content)
    
    return True, "Reaction text stripped from file."

def delete_last_faction_reaction(profile, faction):
    db.delete_last_faction_reaction(profile, faction)

def generate_reaction_for_scene(profile_name, filename, faction, public_only=False, format_style="Standard", custom_instructions=""):
    """
    Simulates a faction's reaction (Adaptive Formats).
    Includes 'Format Adaptation' to transmute anachronistic requests (e.g., 'Twitter' in 1200 AD -> 'Town Crier').
    """
    # 1. Alias Resolution (Uses local backend helper)
    true_faction = resolve_faction_alias(profile_name, faction)
    print(f"  [Identity] Resolved '{faction}' -> '{true_faction}'")

    # 2. Retrieve Global Context
    rules, plan, _ = get_global_context(profile_name)
    state = db.get_world_state(profile_name)
    content = db.read_file_content(profile_name, filename)
    settings = db.get_story_settings(profile_name)
    
    # 3. Era / Tech-Level Detection
    use_time = settings.get('use_time_system', 'true').lower() == 'true'
    era_display = "Undefined (Infer Tech Level from Lore)"
    if use_time and state.get('year', 0) > 0:
        era_display = f"{state['year']}"

    # 4. Retrieve Faction Voice (DB Call)
    past_reactions = db.get_recent_faction_memory(profile_name, true_faction)
    
    # 5. Smart Retrieval (Context for the Faction)
    query = f"Faction '{true_faction}' reacting to scene content: {content[:800]}..."
    relevant_ids = get_relevant_fragment_ids(
        profile_name, 
        user_query=query, 
        doc_types=["Lore", "Fact", "Rulebook", "Scene"]
    )
    smart_facts = db.get_content_by_ids(profile_name, relevant_ids)
    
    # 6. Content Sanitization (Fog of War)
    if public_only:
        pattern = r"\[\[PRIVATE\]\].*?\[\[/PRIVATE\]\]"
        content = re.sub(pattern, "[...INTERNAL/PRIVATE SCENE REDACTED...]", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"\[PRIVATE:.*?\]", "[REDACTED]", content)

    # 7. Context Instruction Layer
    knowledge_instr = (
        "You are reading the unredacted scene. HOWEVER, act strictly as the Target Faction. "
        "DO NOT reference internal thoughts of others unless the provided Rules/Lore explicitly grant telepathic abilities. "
        "Otherwise, react ONLY to observable actions."
    )
    if public_only:
        knowledge_instr = (
            "CRITICAL: You are an EXTERNAL OBSERVER. Private interactions have been REDACTED. "
            "You DO NOT know what happened in the redacted sections. Do NOT guess accurately."
        )

    # 8. Prompt Construction (ADAPTIVE FORMATS)
    prompt = f"""
    ROLE: Narrative Simulator (Grounded in History & State).
    TARGET FACTION: {true_faction}
    CURRENT YEAR/ERA: {era_display}
    
    *** WORLD STATE & DATA ***
    Character Roster: {json.dumps(state.get('Cast', []))}
    
    *** RELEVANT INTELLIGENCE (SMART RETRIEVAL) ***
    {smart_facts}
    
    *** VOICE & TONE REFERENCE ***
    {past_reactions}

    *** MISSION ***
    Write a reaction to the SCENE provided below.

    *** FORMAT ADAPTATION PROTOCOL (CRITICAL) ***
    Requested Format: "{format_style}"
    
    INSTRUCTION: You must check if the Requested Format exists in the Current Era ({era_display}).
    1. IF COMPATIBLE: Use the format as requested (e.g. "Newspaper" in 1920).
    2. IF ANACHRONISTIC: Transmute the format to the closest era-appropriate equivalent.
       - Example: User asks for "Twitter/X" in 1200 AD -> You write "Tavern Gossip" or "Town Square Shouting."
       - Example: User asks for "Newspaper" in 2200 AD -> You write "Holographic News Feed."
       - Example: User asks for "Boardroom Meeting" for a Gang -> You write "Backroom Deal."
    
    *** HIERARCHY OF TRUTH ***
    1. LORE PRIORITY: If Lore says "Telepathy exists," then "Mental Chat" is a valid format.
    2. REALISM: Use real-world logic for the Era to determine how news travels (Horse? Telegraph? Subspace?).

    *** ADDITIONAL INSTRUCTIONS ***
    {custom_instructions if custom_instructions else "Follow standard personality and lore."}
    
    *** KNOWLEDGE CONSTRAINTS ***
    {knowledge_instr}
    
    *** SCENE CONTEXT ***
    {content}
    """
    
    llm = get_llm(profile_name, "reaction")
    res = llm.invoke([HumanMessage(content=prompt)]).content
    
    if "REFUSAL" in res: return False, res

    # 9. Save to Memory (DB)
    db.save_faction_reaction(profile_name, true_faction, res, filename)

    # 10. Append to File
    paths = db.get_paths(profile_name)
    clean_style = format_style.split("->")[-1].strip()
    header = f"\n\n>>> REACTION: {true_faction} | {clean_style} <<<\n"
    
    with open(os.path.join(paths['output'], filename), "a", encoding="utf-8") as f:
        f.write(header + res + "\n")

    # 11. [NEW: SYNC TO DATABASE]
    # Read the updated file content (which now includes the appended reaction)
    # and update the Scene Database so the Smart Search can read the reaction later.
    with open(full_filepath, "r", encoding="utf-8") as f:
        full_updated_content = f.read()
        
    db.upsert_scene(profile_name, filename, full_updated_content)
        
    return True, res

# ==========================================
# 11. COMPILER MODULE
# ==========================================

def compile_manuscript(profile_name, files):
    """Compiles selected files into a single manuscript."""
    return "\n***\n".join([db.read_file_content(profile_name, f) for f in files])

def compile_formatted_manuscript(profile_name: str, selected_files: List[str]) -> Dict[str, bytes]:
    """
    Compiles selected scene files into professional PDF and EPUB formats.
    Performs 'Typesetting' cleanups:
    - Removes [[PRIVATE]] tags (keeps content).
    - Formats 'Reactions' as proper Interludes/Dossiers.
    - Sanitizes smart quotes/dashes for PDF compatibility.
    """
    
    # --- HELPER: TEXT CLEANER ---
    def clean_manuscript_text(text):
        # Remove System Tags (Privacy)
        text = text.replace("[[PRIVATE]]", "").replace("[[/PRIVATE]]", "")
        
        # Sanitize Smart Characters for PDF (Latin-1 safe)
        replacements = {
            '\u201c': '"', '\u201d': '"',  # Smart double quotes
            '\u2018': "'", '\u2019': "'",  # Smart single quotes
            '\u2013': '-', '\u2014': '--', # Dashes
            '\u2026': '...',               # Ellipsis
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
            
        return text

    # --- HELPER: REACTION FORMATTER ---
    def format_reaction_blocks(text):
        """
        Converts raw '>>> REACTION' blocks into stylish 'Interludes'.
        Removes the '✨ Custom' debug lines and metadata clutter.
        """
        # Split the text into the Main Scene and appended Reactions
        parts = re.split(r'>>> REACTION:', text)
        
        # Part 0 is the main story
        final_text = clean_manuscript_text(parts[0].strip())
        
        # Process any reactions (Parts 1+)
        if len(parts) > 1:
            for raw_reaction in parts[1:]:
                # Regex to parse the header: "Faction | Type <<<"
                header_match = re.search(r'\s*(.*?) \|\s*(.*?) <<<', raw_reaction)
                
                if header_match:
                    faction = header_match.group(1).strip()
                    r_type = header_match.group(2).strip()
                    
                    # Remove the header line AND the following "✨" line if it exists
                    body = re.sub(r'\s*(.*?) \|\s*(.*?) <<<(\n✨.*)?', '', raw_reaction, count=1).strip()
                    
                    # Kill the "Category -> Format" line
                    body = re.sub(r'^.* -> .*$', '', body, flags=re.MULTILINE)
                    
                    # Kill the "PARTIES:" line entirely
                    body = re.sub(r'^\s*(\*\*)?PARTIES:.*$', '', body, flags=re.MULTILINE)
                    
                    # Remove double stars (**Name**) if they exist in the body
                    body = re.sub(r'\*\*(.*?)\*\*', r'\1', body)

                    # Clean up extra empty lines created by the deletions
                    body = re.sub(r'\n{3,}', '\n\n', body).strip()

                    # Sanitize body text
                    body = clean_manuscript_text(body)
                    
                    # PDF/EPUB visual separator
                    final_text += "\n\n" + ("*" * 20) + "\n\n" 
                    final_text += f"INTERLUDE: {faction.upper()}\n"
                    final_text += f"Type: {r_type}\n"
                    final_text += ("-" * 20) + "\n\n"
                    final_text += body
                else:
                    # Fallback if regex fails (just clean and append)
                    final_text += "\n\n***\n\n" + clean_manuscript_text(raw_reaction)
                    
        return final_text

    # --- DATA AGGREGATION ---
    chapters = []
    for filename in selected_files:
        # Use DB Manager to read file content
        raw_content = db.read_file_content(profile_name, filename)
        
        # Apply the Typesetting Logic
        formatted_body = format_reaction_blocks(raw_content)

        # Smart Title Logic (Ch04 -> Chapter 4)
        base_name = filename.replace(".txt", "")
        chapter_prefix = ""
        match = re.search(r'(Ch\d+|Chapter_\d+)', base_name, re.IGNORECASE)
        if match:
            try:
                num = int(re.search(r'\d+', match.group(0)).group(0))
                chapter_prefix = f"Chapter {num}: "
            except: pass

        clean_parts = [p for p in base_name.split("_") if not re.match(r'(Ch\d+|Chapter|\d{4})', p)]
        raw_title = " ".join(clean_parts)
        final_title = f"{chapter_prefix}{raw_title}"

        chapters.append({"title": final_title, "body": formatted_body})

    results = {"pdf": None, "epub": None}

    # --- PDF PIPELINE (fpdf2) ---
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Front Matter
        pdf.add_page()
        pdf.set_font("Times", "B", 24)
        pdf.cell(0, 60, f"Story Profile: {profile_name}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Times", "", 12)
        pdf.cell(0, 10, "Generated by Chronos Story Director", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.add_page()

        # Chapter Loop
        for chap in chapters:
            pdf.set_font("Times", "B", 16)
            pdf.cell(0, 10, chap['title'], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)
            
            pdf.set_font("Times", "", 12)
            # Encode/Decode to handle Latin-1 limitations of standard FPDF
            safe_body = chap['body'].encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 6, safe_body)
            pdf.ln(10) 
            pdf.add_page()

        results["pdf"] = bytes(pdf.output())
    except Exception as e:
        print(f"PDF Generation Error: {e}")

    # --- EPUB PIPELINE (EbookLib) ---
    try:
        book = epub.EpubBook()
        book.set_identifier(profile_name)
        book.set_title(profile_name)
        book.set_language('en')

        epub_chapters = []
        for i, chap in enumerate(chapters):
            c = epub.EpubHtml(title=chap['title'], file_name=f'chap_{i}.xhtml', lang='en')
            html_body = chap['body'].replace("\n", "<br/>")
            html_body = html_body.replace("********************", "<hr/>")
            
            c.content = f"<h1>{chap['title']}</h1><p>{html_body}</p>"
            book.add_item(c)
            epub_chapters.append(c)

        book.toc = (epub_chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        style = 'body { font-family: serif; } h1 { text-align: center; }'
        nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
        book.add_item(nav_css)
        book.spine = ['nav'] + epub_chapters
        
        buffer = BytesIO()
        epub.write_epub(buffer, book, {})
        buffer.seek(0)
        results["epub"] = buffer.getvalue()
    except Exception as e:
        print(f"EPUB Generation Error: {e}")

    return results