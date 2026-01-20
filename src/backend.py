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
import smart_retrieval
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
    chapter_num: int
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

def get_next_chapter_number(profile_name: str) -> int:
    """Scans existing files to find the next available chapter number."""
    paths = get_paths(profile_name)
    files = glob.glob(os.path.join(paths['output'], "*.txt"))
    max_ch = 0
    for f in files:
        base = os.path.basename(f)
        # Regex to find 'Ch01' or 'Chapter_1'
        match = re.search(r'Ch(?:apter)?_?(\d+)', base, re.IGNORECASE)
        if match:
            try:
                num = int(match.group(1))
                if num > max_ch: max_ch = num
            except: pass
    return max_ch + 1

def get_global_context(profile_name: str):
    """
    Retrieves the 'Immutable' context layers that must be present in every generation cycle.
    1. Rules: The physics/magic/laws of the world.
    2. Plan: The strategic direction of the story.
    3. Spoilers: Critical secrets to protect.
    """
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    
    # World Rules (Always Active)
    c.execute("SELECT content FROM memory_fragments WHERE type='Rulebook'")
    rules = "\n\n".join([r[0] for r in c.fetchall()])
    
    # Strategic Plan (Always Active)
    c.execute("SELECT content FROM memory_fragments WHERE type='Plan' LIMIT 1")
    row = c.fetchone()
    plan = row[0] if row else "NO PLAN ESTABLISHED."
    
    # Spoilers (Global Safety)
    c.execute("SELECT content FROM memory_fragments WHERE type='Spoiler'")
    spoilers = [r[0] for r in c.fetchall()]
    
    conn.close()
    return rules, plan, spoilers

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
        "model_reaction": "gemini-2.5-flash",
        "model_retrieval": "gemini-2.5-flash"
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
        "reaction": "model_reaction",
        "analysis": "model_analysis",
        "retrieval": "model_retrieval"
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

    # Faction Memory
    c.execute('''CREATE TABLE IF NOT EXISTS faction_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        faction_name TEXT,
        reaction_text TEXT,
        source_scene TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

def update_story_setting(profile_name: str, key: str, value: str):
    """Upserts a global configuration setting."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO story_settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

def get_content_by_ids(profile_name, id_list):
    """Retrieves full text content for a specific list of fragment IDs."""
    if not id_list: return ""
    
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    
    # Safe parameterized query for a list
    placeholders = ','.join(['?'] * len(id_list))
    query = f"SELECT content FROM memory_fragments WHERE id IN ({placeholders})"
    
    try:
        c.execute(query, id_list)
        rows = c.fetchall()
        return "\n\n".join([r[0] for r in rows])
    except Exception as e:
        print(f"Fetch Error: {e}")
        return ""
    finally:
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

def rename_fragment(profile_name, frag_id, new_filename):
    """Updates the display label/filename of an existing fragment."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("UPDATE memory_fragments SET source_filename = ? WHERE id = ?", (new_filename, frag_id))
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

def add_project(profile_name, name, description, features):
    """Initialize a new tracked project in the world state."""
    state = get_world_state(profile_name)
    if "Projects" not in state: state["Projects"] = []
    new_proj = {"Name": name, "Description": description, "Features_Specs": features, "Progress": 0}
    state["Projects"].append(new_proj)
    save_world_state(profile_name, state)

def update_project(profile_name, project_index, progress, notes, new_name=None, new_desc=None):
    """Updates progress, specifications, name, and description for an existing project."""
    state = get_world_state(profile_name)
    if "Projects" in state and 0 <= project_index < len(state["Projects"]):
        state["Projects"][project_index]["Progress"] = progress
        if notes is not None: state["Projects"][project_index]["Features_Specs"] = notes
        if new_name is not None: state["Projects"][project_index]["Name"] = new_name
        if new_desc is not None: state["Projects"][project_index]["Description"] = new_desc
        
        save_world_state(profile_name, state)

def complete_project(profile_name, project_index, custom_lore_text, target_category="Fact"):
    """
    Archives a completed project and converts it into a permanent historical fragment.
    """
    state = get_world_state(profile_name)
    if "Projects" in state and 0 <= project_index < len(state["Projects"]):
        proj = state["Projects"][project_index]
        entry_title = f"PROJECT: {proj['Name']}"

        add_fragment(profile_name, entry_title, custom_lore_text, target_category)

        del state["Projects"][project_index]
        save_world_state(profile_name, state)
        return True, f"Project '{proj['Name']}' archived to {target_category}."
    return False, "Project not found."

def analyze_state_changes(profile_name, scene_content):
    """
    Executes an LLM analysis of the scene to auto-update the world state (JSON).
    Detects changes in allies, assets, skills, reputation, AND abstract World Variables.
    """
    state = get_world_state(profile_name)
    
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

    3. ALLIES: Update Loyalty (0-100) or Status if changed. Add new major characters.

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
        if new_state:
            if "Protagonist Status" not in new_state:
                state.update(new_state)
                return state
            else:
                return new_state
        return state

    except Exception as e: 
        print(f"Analysis Error: {e}")
        return state

# --- SCENE GENERATION WORKFLOW ---

def draft_scene(state: StoryState):
    """
    Workflow Node 1: Narrative Drafting (Adaptive Realism).
    """
    profile = state['profile_name']
    brief = state['scene_brief']
    settings = get_story_settings(profile)
    state_tracking = get_world_state(profile)
    
    # Retrieve Global Context
    rules, plan, db_spoilers = get_global_context(profile)
    
    # Smart Retrieval
    print(f"  [Librarian] Scanning Knowledge Base for: '{brief[:50]}...'")
    relevant_ids = smart_retrieval.get_relevant_fragment_ids(
        profile, 
        user_query=brief, 
        doc_types=["Lore", "Fact", "Rulebook"]
    )
    
    smart_context_str = get_content_by_ids(profile, relevant_ids)
    if not smart_context_str:
        smart_context_str = "No specific historical records found for this scene."

    # Dynamic Spoiler Injection
    dynamic_spoilers = extract_dynamic_spoilers(plan, state['year'], profile, settings=settings) 
    all_banned = list(set(db_spoilers + dynamic_spoilers))
    
    # Header & Era Detection
    use_time_system = settings.get('use_time_system', 'true').lower() == 'true'
    header = ""
    era_display = "Undefined (Infer Tech Level from Lore)"
    
    if use_time_system and state['year'] > 0:
        header = f"{state['date_str']}, {state['year']}"
        era_display = f"{state['year']}"
        if state['time_str']: header += f"\n{state['time_str']}"

    # Timeline Logic
    timeline_section = ""
    if settings.get('use_timelines', 'true').lower() == 'true':
        timelines_list = state_tracking.get("Timelines", [])
        if timelines_list:
            timeline_section = "ACTIVE TIMELINES (MULTIVERSE):\n"
            for t in timelines_list:
                timeline_section += f"- {t.get('Name', 'Unknown')}: {t.get('Description', '')}\n"
    
    # World Variables
    variables_section = ""
    world_vars = state_tracking.get("World Variables", [])
    if world_vars:
        variables_section = "*** WORLD MECHANICS (STRICT) ***\n"
        for v in world_vars:
            variables_section += f"- {v.get('Name', 'Var')}: {v.get('Value', '0')} (RULE: {v.get('Mechanic', '')})\n"

    # Privacy Protocol
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

    # Construct Final Prompt (ADAPTIVE REALISM)
    prompt = f"""
    ROLE: Novelist (Third Person Limited).
    CHARACTER: {settings['protagonist']}.
    CHAPTER: {state['chapter_num']}
    CURRENT CALENDAR YEAR: {era_display}
    
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
    
    # Execute Generation
    llm = get_llm(profile, "scene", settings=settings)
    response = llm.invoke([HumanMessage(content=prompt)]).content
    
    return {
        "current_draft": response, 
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

def generate_scene(profile_name, chapter_num, year, date_str, time_str, title, brief, context_files_list=None, use_fog_of_war=False):
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
        "chapter_num": chapter_num,
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

    # Format: Ch01_1984-03-05_The_Title.txt
    prefix = f"Ch{int(chapter_num):02d}"
    
    if use_time:
        safe_date = final_date.replace(" ", "-")
        filename = f"{prefix}_{final_year}-{safe_date}_{safe_title}.txt"
    else:
        filename = f"{safe_title}.txt"
    
    paths = get_paths(profile_name)
    filepath = os.path.join(paths['output'], filename)
    
    # FALLBACK LOGIC: If the user left Chapter or Year empty (None), use the hints
    if chapter_num is None:
        chapter_num = get_next_chapter_number(profile_name)
    
    if year is None:
        year = get_world_state(profile_name).get("Current_Year", None)

    # Collision Avoidance
    counter = 1
    while os.path.exists(filepath):
        # Handle duplicates for both formats
        if use_time:
            filename = f"{prefix}_{final_year}-{safe_date}_{safe_title}_{counter}.txt"
        else:
            filename = f"{prefix}_{safe_title}_{counter}.txt"
            
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
    """
    Interacts with the Co-Author persona (Adaptive Logic).
    """
    # Retrieve Global Context
    rules, plan, _ = get_global_context(profile_name)
    state = get_world_state(profile_name)
    settings = get_story_settings(profile_name)

    # Get Recent Narrative
    recent_scenes = get_last_scenes(profile_name)
    
    # Smart Retrieval
    print(f"  [Co-Author] Researching: '{user_input[:50]}...'")
    relevant_ids = smart_retrieval.get_relevant_fragment_ids(
        profile_name, 
        user_query=user_input, 
        doc_types=["Lore", "Fact", "Rulebook"]
    )
    
    smart_knowledge = get_content_by_ids(profile_name, relevant_ids)
    if not smart_knowledge:
        smart_knowledge = "No specific database records found (Relying on General Knowledge)."

    # Era / Tech-Level Detection (Standardized)
    use_time_system = settings.get('use_time_system', 'true').lower() == 'true'
    era_display = "Undefined (Infer Tech Level from Lore)"
    if use_time_system and state.get('year', 0) > 0:
        era_display = f"{state['year']}"
    
    # Construct Prompt (HIERARCHY OF TRUTH + CALENDAR AGNOSTIC)
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

def generate_reaction_for_scene(profile_name, filename, faction, public_only=False, format_style="Standard", custom_instructions=""):
    """
    Simulates a faction's reaction (Adaptive Formats).
    
    Upgrade: Includes 'Format Adaptation'. If the user requests 'Social Media' 
    in a Medieval setting, the AI automatically transmutes it to 'Tavern Gossip' 
    or 'Town Crier' to preserve immersion.
    """
    # Alias Resolution
    true_faction = smart_retrieval.resolve_faction_alias(profile_name, faction)
    print(f"  [Identity] Resolved '{faction}' -> '{true_faction}'")

    # Retrieve Global Context
    rules, plan, _ = get_global_context(profile_name)
    state = get_world_state(profile_name)
    content = read_file_content(profile_name, filename)
    settings = get_story_settings(profile_name)
    
    # Era / Tech-Level Detection (Same as Draft Scene)
    use_time_system = settings.get('use_time_system', 'true').lower() == 'true'
    era_display = "Undefined (Infer Tech Level from Lore)"
    if use_time_system and state.get('year', 0) > 0:
        era_display = f"{state['year']}"

    # Retrieve Faction Voice
    past_reactions = get_recent_faction_memory(profile_name, true_faction)
    
    # Smart Retrieval
    query = f"Faction '{true_faction}' reacting to scene content: {content[:800]}..."
    relevant_ids = smart_retrieval.get_relevant_fragment_ids(
        profile_name, 
        user_query=query, 
        doc_types=["Lore", "Fact", "Rulebook"]
    )
    smart_facts = get_content_by_ids(profile_name, relevant_ids)
    
    # Content Sanitization (Fog of War)
    if public_only:
        pattern = r"\[\[PRIVATE\]\].*?\[\[/PRIVATE\]\]"
        content = re.sub(pattern, "[...INTERNAL/PRIVATE SCENE REDACTED...]", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"\[PRIVATE:.*?\]", "[REDACTED]", content)

    # Context Instruction Layer
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

    # Prompt Construction (ADAPTIVE FORMATS)
    prompt = f"""
    ROLE: Narrative Simulator (Grounded in History & State).
    TARGET FACTION: {true_faction}
    CURRENT YEAR/ERA: {era_display}
    
    *** WORLD STATE & DATA ***
    Character Data: {json.dumps(state.get('Allies', []))}
    
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

    save_faction_reaction(profile_name, true_faction, res, filename)

    paths = get_paths(profile_name)
    clean_style = format_style.split("->")[-1].strip()
    header = f"\n\n>>> REACTION: {true_faction} | {clean_style} <<<\n"
    
    with open(os.path.join(paths['output'], filename), "a", encoding="utf-8") as f:
        f.write(header + res + "\n")
        
    return True, res

def save_faction_reaction(profile_name, faction, text, scene_name):
    """Logs a raw reaction to the database to preserve 'Voice' and 'Tone'."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    c.execute("INSERT INTO faction_memory (faction_name, reaction_text, source_scene) VALUES (?, ?, ?)", 
              (faction, text, scene_name))
    conn.commit()
    conn.close()

def get_recent_faction_memory(profile_name, faction, limit=3):
    """Retrieves the last few raw reactions to use as style references."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    try:
        c.execute("""
            SELECT reaction_text, source_scene FROM faction_memory 
            WHERE faction_name = ? 
            ORDER BY id DESC LIMIT ?
        """, (faction, limit))
        rows = c.fetchall()
    except sqlite3.OperationalError:
        return "No memory bank found."
    finally:
        conn.close()
    
    if not rows:
        return "No previous records found."
    
    history = ""
    for r in rows:
        history += f"--- FROM SCENE: {r[1]} ---\n{r[0][:800]}...\n\n" 
    return history

def run_war_room_simulation(profile_name, action_input):
    """
    Executes a Monte Carlo strategic simulation (Smart Retrieval).
    
    Upgrade: Now uses the 'Librarian' to find specific historical precedents, 
    enemy capabilities (Lore), and reads recent scenes to understand the 
    immediate tactical situation.
    """
    # Retrieve Global Rules & Plan
    rules, plan, _ = get_global_context(profile_name)
    state = get_world_state(profile_name)
    
    # Get Immediate Tactical Context
    recent_history = get_last_scenes(profile_name)
    
    # Smart Retrieval (Strategic Intelligence)
    print(f"  [War Room] Gathering Intelligence for: '{action_input[:50]}...'")
    relevant_ids = smart_retrieval.get_relevant_fragment_ids(
        profile_name, 
        user_query=f"Strategic analysis of: {action_input}", 
        doc_types=["Lore", "Fact", "Rulebook"]
    )
    smart_intel = get_content_by_ids(profile_name, relevant_ids)
    if not smart_intel:
        smart_intel = "No specific intelligence dossiers found."

    # Construct the Dossier
    intel_packet = f"""
    *** CURRENT ASSETS & STATUS ***
    Protagonist Status: {json.dumps(state.get('Protagonist Status', {}))}
    Known Allies & Enemies: {json.dumps(state.get('Allies', []))}
    Available Assets: {json.dumps(state.get('Assets', []))}
    Current Skills: {json.dumps(state.get('Skills', []))}
    
    *** IMMEDIATE CONTEXT (RECENT EVENTS) ***
    {recent_history[-4000:]} 
    
    *** RELEVANT KNOWLEDGE (LORE/FACTS) ***
    {smart_intel}
    """
    
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
    
    llm = get_llm(profile_name, "analysis")
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

def undo_last_reaction_text(profile_name, filename, faction):
    """
    Removes the last appended reaction for a specific faction from the text file.
    SAFEGUARD: Only deletes if the file explicitly ends with a reaction block for this faction.
    """
    paths = get_paths(profile_name)
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
        
    return True, "Reaction text stripped from file."

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
    Performs 'Typesetting' cleanups:
    - Removes [[PRIVATE]] tags (keeps content).
    - Formats 'Reactions' as proper Interludes/Dossiers.
    - Sanitizes smart quotes/dashes for PDF compatibility.
    """
    paths = get_paths(profile_name)
    
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
        Removes the '✨ Custom' debug lines and metadata clutter (Parties, Format arrows).
        """
        # Split the text into the Main Scene and appended Reactions
        parts = re.split(r'>>> REACTION:', text)
        
        # Part 0 is the main story
        final_text = clean_manuscript_text(parts[0].strip())
        
        # Process any reactions (Parts 1+)
        if len(parts) > 1:
            for raw_reaction in parts[1:]:
                # Regex to parse the header: "Faction | Type <<<"
                # pattern:  " Charles Koch | Private Discussions <<<\n✨ Custom..."
                header_match = re.search(r'\s*(.*?) \|\s*(.*?) <<<', raw_reaction)
                
                if header_match:
                    faction = header_match.group(1).strip()
                    r_type = header_match.group(2).strip()
                    
                    # Remove the header line AND the following "✨" line if it exists
                    body = re.sub(r'\s*(.*?) \|\s*(.*?) <<<(\n✨.*)?', '', raw_reaction, count=1).strip()
                    
                    # Kill the "Category -> Format" line (e.g. "Private Discussion -> Private Conversation")
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
        raw_content = read_file_content(profile_name, filename)
        
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