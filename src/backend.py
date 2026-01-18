"""
Chronos Story Director - Backend Engine
=======================================
Core logic for the Retrieval-Augmented Generation (RAG) storytelling system.
Handles LLM orchestration, state management, database persistence, and 
narrative workflow execution.

Copyright (c) 2025 SirTonyEdgar
Licensed under the MIT License.
"""

import os
import re
import glob
import json
import sqlite3
import shutil
import datetime
from io import BytesIO
from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv

# Third-Party Dependencies
from google import genai as new_genai
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from fpdf import FPDF
from ebooklib import epub

# Optional OpenAI Integration
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- TYPE DEFINITIONS ---
class StoryState(TypedDict):
    """
    State schema for the scene generation workflow.
    Tracks context, configuration, and generation artifacts across graph nodes.
    """
    profile_name: str 
    year: int
    date_str: str
    time_str: str 
    scene_title: str
    scene_brief: str
    current_draft: str
    critique_notes: str
    revision_count: int
    is_grounded: bool
    recent_context: str 
    banned_words: str
    use_fog_of_war: bool

# --- HELPER FUNCTIONS ---

def save_world_state(profile_name: str, new_state_dict: Dict):
    paths = get_paths(profile_name)
    try:
        if os.path.exists(paths['state']):
            backup_name = f"world_state_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join(paths['data'], backup_name)
            shutil.copy(paths['state'], backup_path)
            
            backups = sorted(glob.glob(os.path.join(paths['data'], "world_state_backup_*.json")))
            for old_b in backups[:-5]:
                os.remove(old_b)

        with open(paths['state'], 'w') as f: 
            json.dump(new_state_dict, f, indent=4)
        return True, "State Saved"
    except Exception as e: 
        return False, str(e)

def _extract_json(text: str) -> Dict:
    """
    Robustly extracts JSON objects from LLM responses, handling potential 
    markdown formatting or conversational preamble.
    """
    try:
        # Attempt direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        # Regex fallback to find the first JSON-like structure
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}

class MockResponse:
    """Simulation of an LLM response object for fallback scenarios."""
    def __init__(self, text):
        self.content = text

class MockLLM:
    """Fallback client that returns system error messages instead of raising exceptions."""
    def invoke(self, *args, **kwargs):
        return MockResponse("⚠️ SYSTEM ERROR: API Key missing or invalid configuration.")

# --- INITIALIZATION & CONFIGURATION ---

def get_story_settings(profile_name: str) -> dict:
    """Retrieves global configuration settings from the database."""
    defaults = {
        "protagonist": "The Protagonist",
        "default_timezone": "CST",
        "use_time_system": "true",
        "use_timelines": "true",
        "timeline_a_name": "Timeline Alpha", "timeline_a_desc": "Past",
        "timeline_b_name": "Timeline Prime", "timeline_b_desc": "Present",
        "model_scene": "gemini-2.5-pro",
        "model_chat": "gemini-2.5-flash",
        "model_reaction": "gemini-2.5-flash"
    }
    paths = get_paths(profile_name)
    try:
        conn = sqlite3.connect(paths['db'])
        c = conn.cursor()
        c.execute("SELECT key, value FROM story_settings")
        for k, v in c.fetchall():
            defaults[k] = v
        conn.close()
    except sqlite3.Error:
        pass 
    return defaults

def get_llm(profile_name: str, task_type: str = "scene", settings: Optional[dict] = None):
    """
    Factory method for initializing LLM clients.
    Implements routing logic based on user settings and available API keys.
    """
    if settings is None:
        settings = get_story_settings(profile_name)
    
    # Map task type to setting key
    model_map = {
        "scene": "model_scene",
        "chat": "model_chat",
        "reaction": "model_reaction"
    }
    target_key = model_map.get(task_type, "model_chat")
    model_name = settings.get(target_key, "gemini-2.5-flash")
    
    # Provider detection
    is_gemini = "gemini" in model_name.lower()
    is_gpt = "gpt" in model_name.lower() or "o1" in model_name.lower()

    # 1. Google Provider
    if is_gemini and GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY)
    
    # 2. OpenAI Provider
    if is_gpt and OPENAI_API_KEY and ChatOpenAI:
        return ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)

    # 3. Cross-Provider Fallbacks
    if is_gemini and not GOOGLE_API_KEY and OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    
    if is_gpt and not OPENAI_API_KEY and GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

    # 4. Universal Fallback
    if GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
        
    return MockLLM()

def list_available_models_all() -> List[str]:
    """Retrieves available models using the new Google GenAI Client."""
    models = []
    
    if GOOGLE_API_KEY:
        try:
            client = new_genai.Client(api_key=GOOGLE_API_KEY)
            for m in client.models.list():
                if "generateContent" in m.supported_generation_methods:
                    models.append(m.name.replace("models/", ""))
        except Exception: 
            pass
        # Hardcoded fallbacks in case API fails
        models.extend(["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"])
    
    if OPENAI_API_KEY:
        models.extend(["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini"])
        
    return sorted(list(set(models)))

# --- FILE SYSTEM & PROFILE MANAGEMENT ---

def get_paths(profile_name: str) -> Dict[str, str]:
    """Constructs absolute file paths for profile resources."""
    root = os.path.join(PROFILES_DIR, profile_name)
    return {
        "root": root,
        "data": os.path.join(root, "data"),
        "db": os.path.join(root, "data", "story_database.db"),
        "state": os.path.join(root, "data", "world_state.json"),
        "output": os.path.join(root, "output", "scenes"),
        "lore": os.path.join(root, "input", "lore")
    }

def ensure_profile_structure(profile_name: str):
    """Initializes the directory structure and database for a new profile."""
    paths = get_paths(profile_name)
    for p in [paths['root'], paths['data'], paths['output'], paths['lore']]:
        os.makedirs(p, exist_ok=True)
    
    init_db(profile_name)
    
    if not os.path.exists(paths['state']):
        default_state = {"Status": "New Game", "Assets": [], "Allies": [], "Projects": []}
        with open(paths['state'], 'w') as f: 
            json.dump(default_state, f, indent=4)
    return paths

def list_profiles() -> List[str]:
    """Enumerates available user profiles."""
    if not os.path.exists(PROFILES_DIR): return []
    return [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d))]

# --- DATABASE PERSISTENCE ---

def init_db(profile_name: str):
    """Initializes SQLite schema for persistent storage."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30) 
    c = conn.cursor()
    
    # RAG Storage (Lore, Plans, Rules)
    c.execute('''CREATE TABLE IF NOT EXISTS memory_fragments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_filename TEXT, content TEXT, type TEXT, year INTEGER DEFAULT NULL
    )''')
    
    # Configuration Storage
    c.execute('''CREATE TABLE IF NOT EXISTS story_settings (
        key TEXT PRIMARY KEY, value TEXT
    )''')
    
    # Interaction Logs
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def update_story_setting(profile_name: str, key: str, value: str):
    """Upserts a global configuration setting."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO story_settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

# --- RAG & CONTEXT AGGREGATION ---

def get_full_context_data(profile_name: str):
    """Retrieves all relevant context layers (Lore, Rules, Plans) for the prompt."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    
    c.execute("SELECT content FROM memory_fragments WHERE type='Lore'")
    lore = "\n\n".join([r[0] for r in c.fetchall()])
    
    c.execute("SELECT content FROM memory_fragments WHERE type='Rulebook'")
    rules = "\n\n".join([r[0] for r in c.fetchall()])
    
    c.execute("SELECT content FROM memory_fragments WHERE type='Plan' LIMIT 1")
    row = c.fetchone()
    plan = row[0] if row else "NO PLAN."
    
    c.execute("SELECT content FROM memory_fragments WHERE type='Fact'")
    facts = "\n".join([f"- {r[0]}" for r in c.fetchall()])
    
    c.execute("SELECT content FROM memory_fragments WHERE type='Spoiler'")
    spoilers = [r[0] for r in c.fetchall()]
    
    conn.close()
    return lore, rules, plan, facts, spoilers

def get_initial_lore(profile_name: str) -> str:
    """Fallback context provider for the initial session."""
    frags = get_fragments(profile_name, "Lore")
    if frags:
        return f"=== BACKGROUND LORE ===\n{frags[0][2]}"
    return "NO LORE ESTABLISHED. STARTING FRESH."

def get_fragments(profile_name: str, doc_type: Optional[str] = None):
    """Queries memory fragments with optional type filtering."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    if doc_type: 
        c.execute("SELECT id, source_filename, content, type FROM memory_fragments WHERE type = ? ORDER BY id DESC", (doc_type,))
    else: 
        c.execute("SELECT id, source_filename, content, type FROM memory_fragments ORDER BY type, id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def add_fragment(profile_name, filename, content, doc_type):
    """Persists a new document fragment to the database."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    c.execute("INSERT INTO memory_fragments (source_filename, content, type) VALUES (?, ?, ?)", (filename, content, doc_type))
    conn.commit()
    conn.close()

def update_fragment(profile_name, frag_id, new_content):
    """Updates the content of an existing fragment."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("UPDATE memory_fragments SET content = ? WHERE id = ?", (new_content, frag_id))
    conn.commit()
    conn.close()

def delete_fragment(profile_name, frag_id):
    """Removes a fragment from the database."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("DELETE FROM memory_fragments WHERE id = ?", (frag_id,))
    conn.commit()
    conn.close()

# --- WORLD STATE MANAGEMENT ---

def get_world_state(profile_name: str) -> Dict:
    """Reads the current world state JSON."""
    paths = get_paths(profile_name)
    try:
        with open(paths['state'], 'r') as f: 
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_world_state(profile_name: str, new_state_dict: Dict):
    """Writes updates to the world state JSON."""
    paths = get_paths(profile_name)
    try:
        with open(paths['state'], 'w') as f: 
            json.dump(new_state_dict, f, indent=4)
        return True, "State Saved"
    except Exception as e: 
        return False, str(e)

def add_project(profile_name, name, description, features):
    """Initialize a new tracked project in the world state."""
    state = get_world_state(profile_name)
    if "Projects" not in state: state["Projects"] = []
    new_proj = {"Name": name, "Description": description, "Features_Specs": features, "Progress": 0}
    state["Projects"].append(new_proj)
    save_world_state(profile_name, state)

def update_project(profile_name, project_index, progress, notes):
    """Updates progress or specifications for an existing project."""
    state = get_world_state(profile_name)
    if "Projects" in state and 0 <= project_index < len(state["Projects"]):
        state["Projects"][project_index]["Progress"] = progress
        if notes: state["Projects"][project_index]["Features_Specs"] = notes
        save_world_state(profile_name, state)

def complete_project(profile_name, project_index, custom_lore_text):
    """Archives a completed project and converts it into a permanent historical Fact."""
    state = get_world_state(profile_name)
    if "Projects" in state and 0 <= project_index < len(state["Projects"]):
        proj = state["Projects"][project_index]
        fact_title = f"HISTORY: {proj['Name']}"
        add_fragment(profile_name, fact_title, custom_lore_text, "Fact")
        del state["Projects"][project_index]
        save_world_state(profile_name, state)
        return True, f"Project '{proj['Name']}' cemented in history."
    return False, "Project not found."

def analyze_state_changes(profile_name, scene_content):
    """
    Executes an LLM analysis of the scene to auto-update the world state (JSON).
    Detects changes in allies, assets, skills, and reputation.
    """
    state = get_world_state(profile_name)
    
    prompt = f"""
    ROLE: World State Database Manager.
    
    TASK: Analyze the narrative SCENE and update the JSON STATE to reflect changes.
    
    *** CURRENT STATE ***
    {json.dumps(state)}
    
    *** NARRATIVE SCENE ***
    {scene_content}
    
    *** UPDATE INSTRUCTIONS ***
    1. ALLIES: Update Loyalty (0-100) or Status if changed. Add new major characters.
    2. ASSETS: Add new resources/locations gained. Mark lost assets as "Destroyed".
    3. SKILLS: Add new skills learned.
    4. ALIASES & REPUTATION (CRITICAL): 
       - Look for new titles, nicknames, or reputations bestowed upon the protagonist by the public or other characters.
       - Example: If they conquer a city, add "Conqueror of [City]".
       - Example: If they fix the economy, add "The Architect".
       - MERGE these into the existing 'Aliases' string in 'Protagonist Status' (comma-separated).
    
    OUTPUT: Return ONLY the updated JSON.
    """
    
    llm = get_llm(profile_name, "chat") 
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content
        # Use robust extractor instead of simple replace
        return _extract_json(res)
    except Exception as e: 
        print(f"Analysis Error: {e}")
        return state

# --- SCENE GENERATION WORKFLOW ---

def draft_scene(state: StoryState):
    """
    Workflow Node 1: Narrative Drafting.
    Generates the initial scene prose based on aggregated context, laws, and user brief.
    Includes Conditional Privacy (Fog of War) logic if enabled.
    """
    profile = state['profile_name']
    lore, rules, plan, facts, db_spoilers = get_full_context_data(profile)
    settings = get_story_settings(profile)
    state_tracking = get_world_state(profile)
    
    # Dynamic context injection (Spoilers & Banned Words)
    dynamic_spoilers = extract_dynamic_spoilers(plan, state['year'], profile, settings=settings) 
    all_banned = list(set(db_spoilers + dynamic_spoilers))
    
    # Header generation (Date/Time)
    use_time_system = settings.get('use_time_system', 'true').lower() == 'true'
    header = ""
    if use_time_system:
        header = f"{state['date_str']}, {state['year']}"
        if state['time_str']: 
            header += f"\n{state['time_str']}"
    
    # Multiverse/Timeline logic
    timeline_section = ""
    if settings.get('use_timelines', 'true').lower() == 'true':
        timelines_list = state_tracking.get("Timelines", [])
        if timelines_list:
            timeline_section = "ACTIVE TIMELINES (MULTIVERSE):\n"
            for t in timelines_list:
                t_name = t.get("Name", "Unknown")
                t_desc = t.get("Description", "No description")
                timeline_section += f"- {t_name}: {t_desc}\n"
    
    # Conditional Privacy Logic
    privacy_protocol = ""
    if state.get('use_fog_of_war', False):
        privacy_protocol = """
        *** PRIVACY & FOG OF WAR PROTOCOL (CRITICAL) ***
        You are responsible for marking "Secret Information" for the simulation engine.
        
        RULE: Whenever the characters are in a location or context where the GENERAL PUBLIC / MEDIA cannot see or hear them (e.g., inside a private home, a moving car, a secure bunker, a whispered conversation, or internal monologue), you MUST wrap that specific section of text in [[PRIVATE]] ... [[/PRIVATE]] tags.
        
        EXAMPLE:
        The two men walked through the park. "Nice day," JFK said.
        [[PRIVATE]]
        Inside the car, the smile dropped. "We have a problem," he whispered.
        [[/PRIVATE]]
        """

    prompt = f"""
    ROLE: Novelist (Third Person Limited).
    CHARACTER: {settings['protagonist']}.
    
    *** WORLD LAWS & MECHANICS (STRICT) ***
    {rules}
    {privacy_protocol}
    
    *** STORY BIBLE (LORE) ***
    {lore}

    *** WORLD STATE & PROJECTS ***
    {json.dumps(state_tracking)}
    
    *** FORMATTING ***
    {header}
    
    (Start prose below header. NO Title in body. NO 'But/And' starts).
    
    {timeline_section}
    
    *** FACTS ***
    {facts}
    
    *** BANNED ***
    [{", ".join(all_banned)}]
    
    *** CONTEXT ***
    {state['recent_context']}
    
    === MISSION ===
    BRIEF: {state['scene_brief']}
    
    CRITIQUE: {state['critique_notes']}
    """
    
    llm = get_llm(profile, "scene", settings=settings)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_draft": response.content, 
        "revision_count": state['revision_count'] + 1, 
        "banned_words": ", ".join(all_banned)
    }

def critique_scene(state: StoryState):
    """
    Workflow Node 2: Validation.
    Checks adherence to constraints (e.g. banned words) and logic integrity.
    """
    prompt = f"ROLE: Editor. CHECK: Banned [{state['banned_words']}]? DRAFT: {state['current_draft']} OUTPUT: PASS/FAIL"
    llm = get_llm(state['profile_name'], "chat")
    if "PASS" in llm.invoke([HumanMessage(content=prompt)]).content:
        return {"is_grounded": True, "critique_notes": ""}
    return {"is_grounded": False, "critique_notes": "Found banned content or logic break."}

def auto_generate_title(profile_name, draft_text, brief):
    """Generates a short, evocative title based on the generated scene content."""
    prompt = f"""
    TASK: Create a Title.
    SCENE BRIEF: {brief}
    SCENE CONTENT START: {draft_text[:1000]}...
    
    INSTRUCTION: Generate a short, punchy, dramatic title (max 6 words) for this scene. 
    Examples: "The Red Wedding", "Midnight at the Docks", "Protocol Omega".
    OUTPUT: The title text ONLY. No quotes.
    """
    llm = get_llm(profile_name, "chat")
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()

def generate_scene(profile_name, year, date_str, time_str, title, brief, context_files_list=None, use_fog_of_war=False):
    """
    Entry point for the scene generation pipeline.
    Initializes the state graph, aggregates context, and executes the workflow.
    """
    # GRAPH SETUP
    workflow = StateGraph(StoryState)
    workflow.add_node("drafter", draft_scene)
    workflow.add_node("validator", critique_scene)
    workflow.set_entry_point("drafter")
    workflow.add_edge("drafter", "validator")
    workflow.add_conditional_edges("validator", lambda s: END if s['is_grounded'] or s['revision_count'] > 2 else "drafter")
    app = workflow.compile()
    
    # CONTEXT ASSEMBLY
    context_str = ""
    if context_files_list:
        for fname in context_files_list:
            if fname == "Auto (Last 3 Scenes)": 
                context_str += get_last_scenes(profile_name)
            else: 
                context_str += f"\n=== CONTEXT: {fname} ===\n{read_file_content(profile_name, fname)[:5000]}\n"
    else: 
        context_str = get_initial_lore(profile_name)

    settings = get_story_settings(profile_name)
    
    # HEURISTIC TIME INFERENCE
    use_time = settings.get('use_time_system', 'true').lower() == 'true'
    final_year = year
    final_date = date_str
    final_time = time_str
    
    if use_time and (not final_year or not final_date or not final_time):
        inferred = infer_header_data(brief, context_str, settings, profile_name)
        if not final_year: final_year = inferred.get('year', 1984)
        if not final_date: final_date = inferred.get('date', "Unknown Date")
        if not final_time: final_time = inferred.get('time', "")

    try: final_year = int(final_year)
    except: final_year = 0

    # INITIAL STATE
    temp_title = title if title else "Untitled Processing..."

    initial_input = {
        "profile_name": profile_name,
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
        "use_fog_of_war": use_fog_of_war
    }
    
    final_state = app.invoke(initial_input)
    
    # AUTO-TITLE LOGIC
    final_title = title
    if not final_title:
        final_title = auto_generate_title(profile_name, final_state['current_draft'], brief)

    # OUTPUT PERSISTENCE
    safe_title = re.sub(r'[\\/*?:"<>|]', "", final_title).replace(" ", "_")
    
    if use_time:
        safe_date = final_date.replace(" ", "-")
        filename = f"{final_year}-{safe_date}_{safe_title}.txt"
    else:
        filename = f"{safe_title}.txt"
    
    paths = get_paths(profile_name)
    filepath = os.path.join(paths['output'], filename)
    
    # Collision Avoidance
    counter = 1
    while os.path.exists(filepath):
        # Handle duplicates for both formats
        if use_time:
            filename = f"{final_year}-{safe_date}_{safe_title}_{counter}.txt"
        else:
            filename = f"{safe_title}_{counter}.txt"
            
        filepath = os.path.join(paths['output'], filename)
        counter += 1
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_state['current_draft'])
        
    return final_state['current_draft'], filepath

# --- AUXILIARY AI TOOLS ---

def extract_dynamic_spoilers(plan, year, profile_name, settings=None):
    """Parses future events from the 'Plan' to prevent context leakage."""
    if not plan or plan == "NO PLAN.": return []
    prompt = f"List FUTURE events after {year} from: {plan}. OUTPUT: Comma-separated."
    llm = get_llm(profile_name, "chat", settings=settings) 
    return [x.strip() for x in llm.invoke([HumanMessage(content=prompt)]).content.split(',')]

def infer_header_data(brief, prev_context, settings, profile_name):
    """Estimates the narrative date/time for the scene based on recent context."""
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
    except: 
        return {}

def run_chat_query(profile_name, user_input):
    """Interacts with the Co-Author persona, aware of full narrative context."""
    lore, rules, plan, facts, spoilers = get_full_context_data(profile_name)
    state = get_world_state(profile_name)

    recent_scenes = get_last_scenes(profile_name)
    
    prompt = f"""
    ROLE: Co-Author. 
    
    *** STORY BIBLE ***
    LORE: {lore}
    RULES: {rules}
    ESTABLISHED FACTS: {facts}
    FUTURE PLANS: {plan[:3000]}
    
    *** CURRENT STATUS ***
    WORLD STATE: {state}
    RECENT NARRATIVE: {recent_scenes}
    
    *** USER QUERY ***
    {user_input}
    """
    
    llm = get_llm(profile_name, "chat")
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_reaction_for_scene(profile_name, filename, faction, public_only=False, format_style="Standard"):
    """
    Simulates a faction's reaction to a scene.
    Supports 'Fog of War' (redacting private content) and custom formatting styles.
    """
    content = read_file_content(profile_name, filename)
    
    # 1. Content Sanitization (Privacy Filter)
    if public_only:
        # Redact multi-line private blocks
        pattern = r"\[\[PRIVATE\]\].*?\[\[/PRIVATE\]\]"
        content = re.sub(pattern, "[...INTERNAL/PRIVATE SCENE REDACTED...]", content, flags=re.DOTALL | re.IGNORECASE)
        
        # Redact inline private tags
        content = re.sub(r"\[PRIVATE:.*?\]", "[REDACTED]", content)

    # 2. Context Instruction Layer
    knowledge_instr = "You have full knowledge of the scene, including internal thoughts."
    if public_only:
        knowledge_instr = (
            "CRITICAL: You are an EXTERNAL OBSERVER. "
            "The scene text has had private interactions REDACTED. "
            "You DO NOT know what happened in the redacted sections. "
            "You ONLY know what was physically visible in public. "
            "Do NOT guess the redacted content accurately. Speculate wildly or ignore it."
        )

    # 3. Prompt Construction
    prompt = f"""
    ROLE: Narrative Simulator.
    TARGET FACTION: {faction}
    
    *** MISSION ***
    Write a reaction to the SCENE provided below from the perspective of the Target Faction.
    
    *** FORMATTING & TONE ***
    The output must strictly follow this format/medium: 
    "{format_style}"
    
    (Adopt the slang, structure, and limitations of this medium).
    
    *** KNOWLEDGE CONSTRAINTS ***
    {knowledge_instr}
    
    *** SCENE CONTEXT ***
    {content}
    """
    
    llm = get_llm(profile_name, "reaction")
    res = llm.invoke([HumanMessage(content=prompt)]).content
    
    if "REFUSAL" in res: return False, res
    
    # 4. Result Persistence
    paths = get_paths(profile_name)
    timestamp = " (Public)" if public_only else " (Omniscient)"
    clean_style = format_style.split("->")[-1].strip()
    header = f"\n\n>>> REACTION: {faction} | {clean_style}{timestamp} <<<\n"
    
    with open(os.path.join(paths['output'], filename), "a", encoding="utf-8") as f:
        f.write(header + res + "\n")
        
    return True, res

def run_war_room_simulation(profile_name, action_input):
    """
    Executes a Monte Carlo strategic simulation for a proposed action.
    Returns a structured risk/reward analysis report.
    """
    lore, rules, plan, facts, spoilers = get_full_context_data(profile_name)
    state = get_world_state(profile_name)
    
    intel_packet = f"""
    *** CURRENT EMPIRE STATE ***
    Protagonist Status: {json.dumps(state.get('Protagonist Status', {}))}
    Known Allies & Enemies: {json.dumps(state.get('Allies', []))}
    Available Assets: {json.dumps(state.get('Assets', []))}
    Current Skills: {json.dumps(state.get('Skills', []))}
    
    *** ESTABLISHED FACTS ***
    {facts}
    """
    
    prompt = f"""
    ROLE: Strategic Simulation Engine.
    
    *** WORLD RULES & CONSTRAINTS ***
    {rules}
    
    *** INTELLIGENCE PACKET ***
    {intel_packet}
    
    *** LORE CONTEXT ***
    {lore[:3000]} 
    
    *** PROPOSED ACTION ***
    "{action_input}"
    
    *** MISSION ***
    Simulate the consequences of this action. Do not write a story. Write a STRATEGIC REPORT.
    
    *** REPORT FORMAT ***
    ## 📊 Simulation Results
    **Probability of Success:** [0-100%]
    
    ### 1. Direct Consequences (T+0 to T+1 Month)
    * [Immediate Outcome]
    * [Resource Cost]
    
    ### 2. Second-Order Effects (The Ripple)
    * [Unintended side effects on allies/enemies]
    * [Political/Economic shifts]
    
    ### 3. Critical Risks (Blowback)
    * [Who gets angry?]
    * [What could go wrong?]
    
    ### 4. Verdict
    [Go / No-Go recommendation]
    """
    
    llm = get_llm(profile_name, "reaction")
    return llm.invoke([HumanMessage(content=prompt)]).content

# --- UTILITIES & FILE I/O ---

def get_all_files_list(profile_name: str) -> List[str]:
    """Returns a sorted list of all scene and lore files."""
    paths = get_paths(profile_name)
    scenes = glob.glob(os.path.join(paths['output'], "*.txt"))
    lore = glob.glob(os.path.join(paths['lore'], "*.txt"))
    all_files = []
    for f in scenes: all_files.append((os.path.basename(f), f, os.path.getmtime(f)))
    for f in lore: all_files.append((os.path.basename(f), f, os.path.getmtime(f)))
    all_files.sort(key=lambda x: x[2], reverse=True)
    return [x[0] for x in all_files]

def read_file_content(profile_name, filename):
    """Safely reads the content of a file from either Output or Lore directories."""
    paths = get_paths(profile_name)
    p1 = os.path.join(paths['output'], filename)
    if os.path.exists(p1): return open(p1, "r", encoding="utf-8").read()
    p2 = os.path.join(paths['lore'], filename)
    if os.path.exists(p2): return open(p2, "r", encoding="utf-8").read()
    return "Error: File not found."

def save_edited_scene(profile_name, filename, content):
    """Overwrites a scene file with edited content."""
    paths = get_paths(profile_name)
    try:
        with open(os.path.join(paths['output'], filename), "w", encoding="utf-8") as f: 
            f.write(content)
        return True, "Saved"
    except Exception as e: 
        return False, str(e)

def delete_specific_scene(profile_name, filename):
    """Permanently deletes a scene file (Lore files are protected)."""
    paths = get_paths(profile_name)
    p = os.path.join(paths['output'], filename)
    if os.path.exists(p):
        os.remove(p)
        return True, "Deleted"
    return False, "Cannot delete Lore"

def get_last_scenes(profile_name):
    """Retrieves the trailing context (last 3 scenes) for continuity."""
    paths = get_paths(profile_name)
    files = glob.glob(os.path.join(paths['output'], "*.txt"))
    if not files: return "NO SCENES."
    files.sort(key=os.path.getmtime)
    context = ""
    for f in files[-3:]:
        context += f"\n=== PREV: {os.path.basename(f)} ===\n{open(f, 'r', encoding='utf-8').read()[:3000]}\n"
    return context

def compile_manuscript(profile_name, files):
    """Compiles selected files into a single manuscript."""
    return "\n***\n".join([read_file_content(profile_name, f) for f in files])

def compile_formatted_manuscript(profile_name: str, selected_files: List[str]) -> Dict[str, bytes]:
    """
    Compiles selected scene files into professional PDF and EPUB formats.
    """
    paths = get_paths(profile_name)
    
    # Data Aggregation
    chapters = []
    for filename in selected_files:
        content = read_file_content(profile_name, filename)
        
        # --- TITLE CLEANING LOGIC ---
        base_name = filename.replace(".txt", "")
        
        # Input: "1984-March-6th_The_Crash" -> Output: "The Crash"
        if "_" in base_name:
            raw_title = base_name.split("_", 1)[1]
        else:
            raw_title = base_name
            
        clean_title = raw_title.replace("_", " ")

        chapters.append({"title": clean_title, "body": content})

    results = {"pdf": None, "epub": None}

    # --- PDF Rendering Pipeline (fpdf2) ---
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

        # Chapter Rendering
        for chap in chapters:
            # Header
            pdf.set_font("Times", "B", 16)
            pdf.cell(0, 10, chap['title'], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)
            
            # Body
            pdf.set_font("Times", "", 12)
            safe_body = chap['body'].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, safe_body)
            pdf.ln(10) 

        results["pdf"] = bytes(pdf.output())
    except Exception as e:
        print(f"PDF Generation Error: {e}")

    # --- EPUB Rendering Pipeline (EbookLib) ---
    try:
        book = epub.EpubBook()
        book.set_identifier(profile_name)
        book.set_title(profile_name)
        book.set_language('en')

        epub_chapters = []
        for i, chap in enumerate(chapters):
            c = epub.EpubHtml(title=chap['title'], file_name=f'chap_{i}.xhtml', lang='en')
            formatted_body = chap['body'].replace("\n", "<br/><br/>")
            c.content = f"<h1>{chap['title']}</h1><p>{formatted_body}</p>"
            book.add_item(c)
            epub_chapters.append(c)

        # Structure & Navigation
        book.toc = (epub_chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Stylesheet
        style = 'body { font-family: serif; } h1 { text-align: center; }'
        nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
        book.add_item(nav_css)

        book.spine = ['nav'] + epub_chapters
        
        # I/O Buffer
        buffer = BytesIO()
        epub.write_epub(buffer, book, {})
        buffer.seek(0)
        results["epub"] = buffer.getvalue()
        
    except Exception as e:
        print(f"EPUB Generation Error: {e}")

    return results

# --- CHAT PERSISTENCE ---

def get_chat_history(profile_name):
    """Loads chat history from the database."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

def save_chat_message(profile_name, role, content):
    """Appends a message to the persistent chat log."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def clear_chat_history(profile_name):
    """Purges the chat history log."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()