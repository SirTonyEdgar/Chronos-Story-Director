"""
Chronos Story Director
======================
A sophisticated RAG-based storytelling engine that orchestrates LLMs, 
tracks world state, and manages creative workflows.

Copyright (c) 2025 SirTonyEdgar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import re
import glob
import json
import sqlite3
from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv

# Third-Party Imports
import google.generativeai as genai 
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

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

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- TYPE DEFINITIONS ---
class StoryState(TypedDict):
    """
    Represents the state of a scene generation workflow.
    Passed between LangGraph nodes.
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

# --- LLM FACTORY & UTILITIES ---

class MockResponse:
    """Mock object to simulate an LLM response object during failure states."""
    def __init__(self, text):
        self.content = text

class MockLLM:
    """Fallback client that returns error messages instead of raising exceptions."""
    def invoke(self, *args, **kwargs):
        return MockResponse("⚠️ SYSTEM ERROR: API Key missing. Check .env or Settings.")

def get_story_settings(profile_name: str) -> dict:
    """Fetches global story settings from DB, falling back to defaults."""
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
    Factory function that initializes the appropriate LLM client (Google/OpenAI)
    based on profile settings. Includes logic for cross-provider fallbacks.
    """
    if settings is None:
        settings = get_story_settings(profile_name)
    
    # Map task to model setting
    model_map = {
        "scene": "model_scene",
        "chat": "model_chat",
        "reaction": "model_reaction"
    }
    target_key = model_map.get(task_type, "model_chat")
    model_name = settings.get(target_key, "gemini-2.5-flash")
    
    # Provider Routing Logic
    is_gemini = "gemini" in model_name.lower()
    is_gpt = "gpt" in model_name.lower() or "o1" in model_name.lower()

    # 1. Primary Preference: Google
    if is_gemini and GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY)
    
    # 2. Primary Preference: OpenAI
    if is_gpt and OPENAI_API_KEY and ChatOpenAI:
        return ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)

    # 3. Fallbacks (Cross-Provider)
    if is_gemini and not GOOGLE_API_KEY and OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    
    if is_gpt and not OPENAI_API_KEY and GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

    # 4. Universal Fallback
    if GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
        
    return MockLLM()

def list_available_models_all() -> List[str]:
    """Scans API keys to determine which models are available for selection."""
    models = []
    
    if GOOGLE_API_KEY:
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name.replace("models/", ""))
        except: 
            pass
        # Hardcoded specific versions in case list_models() is slow/incomplete
        models.extend(["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"])
    
    if OPENAI_API_KEY:
        models.extend(["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini"])
        
    return sorted(list(set(models)))

# --- PROFILE & PATH MANAGEMENT ---

def get_paths(profile_name: str) -> Dict[str, str]:
    """Generates the file system paths for a specific profile."""
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
    """Creates the necessary directory structure and DB for a new profile."""
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
    if not os.path.exists(PROFILES_DIR): return []
    return [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d))]

# --- DATABASE OPERATIONS ---

def init_db(profile_name: str):
    """Initializes SQLite tables for Fragments, Settings, and Chat History."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30) 
    c = conn.cursor()
    
    # RAG Storage (Lore, Plans, Rules)
    c.execute('''CREATE TABLE IF NOT EXISTS memory_fragments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_filename TEXT, content TEXT, type TEXT, year INTEGER DEFAULT NULL
    )''')
    
    # Key-Value Settings
    c.execute('''CREATE TABLE IF NOT EXISTS story_settings (
        key TEXT PRIMARY KEY, value TEXT
    )''')
    
    # Co-Author Chat Log
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def update_story_setting(profile_name: str, key: str, value: str):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO story_settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

# --- RAG & CONTEXT MANAGEMENT ---

def get_full_context_data(profile_name: str):
    """Aggregates all world-building data (Lore, Rules, Plans) for the prompt context."""
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
    """Retrieves the first lore entry as a fallback context for the very first scene."""
    frags = get_fragments(profile_name, "Lore")
    if frags:
        return f"=== BACKGROUND LORE ===\n{frags[0][2]}"
    return "NO LORE ESTABLISHED. STARTING FRESH."

def get_fragments(profile_name: str, doc_type: Optional[str] = None):
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
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    c.execute("INSERT INTO memory_fragments (source_filename, content, type) VALUES (?, ?, ?)", (filename, content, doc_type))
    conn.commit()
    conn.close()

def update_fragment(profile_name, frag_id, new_content):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("UPDATE memory_fragments SET content = ? WHERE id = ?", (new_content, frag_id))
    conn.commit()
    conn.close()

def delete_fragment(profile_name, frag_id):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("DELETE FROM memory_fragments WHERE id = ?", (frag_id,))
    conn.commit()
    conn.close()

# --- STATE & PROJECT TRACKER ---

def get_world_state(profile_name: str) -> Dict:
    paths = get_paths(profile_name)
    try:
        with open(paths['state'], 'r') as f: 
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_world_state(profile_name: str, new_state_dict: Dict):
    paths = get_paths(profile_name)
    try:
        with open(paths['state'], 'w') as f: 
            json.dump(new_state_dict, f, indent=4)
        return True, "State Saved"
    except Exception as e: 
        return False, str(e)

def add_project(profile_name, name, description, features):
    state = get_world_state(profile_name)
    if "Projects" not in state: state["Projects"] = []
    new_proj = {"Name": name, "Description": description, "Features_Specs": features, "Progress": 0}
    state["Projects"].append(new_proj)
    save_world_state(profile_name, state)

def update_project(profile_name, project_index, progress, notes):
    state = get_world_state(profile_name)
    if "Projects" in state and 0 <= project_index < len(state["Projects"]):
        state["Projects"][project_index]["Progress"] = progress
        if notes: state["Projects"][project_index]["Features_Specs"] = notes
        save_world_state(profile_name, state)

def complete_project(profile_name, project_index, custom_lore_text):
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
    """Uses LLM to detect state changes in the narrative and proposes JSON updates."""
    state = get_world_state(profile_name)
    prompt = f"UPDATE JSON STATE based on SCENE. STATE: {json.dumps(state)}. SCENE: {scene_content}. Return JSON only."
    llm = get_llm(profile_name, "chat") 
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content
        clean = res.replace("```json", "").replace("```", "")
        return json.loads(clean)
    except: 
        return state

# --- SCENE GENERATION (LANGGRAPH WORKFLOW) ---

def draft_scene(state: StoryState):
    """
    Node 1: Drafts the scene content based on context, rules, and world state.
    """
    profile = state['profile_name']
    lore, rules, plan, facts, db_spoilers = get_full_context_data(profile)
    settings = get_story_settings(profile)
    state_tracking = get_world_state(profile)
    
    # Logic for Dynamic Spoilers & Banned Words
    dynamic_spoilers = extract_dynamic_spoilers(plan, state['year'], profile, settings=settings) 
    all_banned = list(set(db_spoilers + dynamic_spoilers))
    
    # Formatting Header
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
        else:
            t_a = settings.get('timeline_a_name', 'Timeline Alpha')
            t_b = settings.get('timeline_b_name', 'Timeline Prime')
            timeline_section = f"TIMELINES: 1. {t_a} (Past/Alt). 2. {t_b} (Present/Main)."
    
    prompt = f"""
    ROLE: Novelist (Third Person Limited).
    CHARACTER: {settings['protagonist']}.
    
    *** WORLD LAWS & MECHANICS (STRICT) ***
    {rules}
    
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
    Node 2: checks constraints (Banned Words) and adherence to logic.
    """
    prompt = f"ROLE: Editor. CHECK: Banned [{state['banned_words']}]? DRAFT: {state['current_draft']} OUTPUT: PASS/FAIL"
    llm = get_llm(state['profile_name'], "chat")
    if "PASS" in llm.invoke([HumanMessage(content=prompt)]).content:
        return {"is_grounded": True, "critique_notes": ""}
    return {"is_grounded": False, "critique_notes": "Found banned content or logic break."}

def generate_scene(profile_name, year, date_str, time_str, title, brief, context_files_list=None):
    """
    Orchestrates the drafting process. Builds the prompt context from files,
    initializes the LangGraph workflow, and handles file saving logic.
    """
    # Build Graph
    workflow = StateGraph(StoryState)
    workflow.add_node("drafter", draft_scene)
    workflow.add_node("validator", critique_scene)
    workflow.set_entry_point("drafter")
    workflow.add_edge("drafter", "validator")
    # Loop back if validation fails, up to 3 tries
    workflow.add_conditional_edges("validator", lambda s: END if s['is_grounded'] or s['revision_count'] > 2 else "drafter")
    app = workflow.compile()
    
    # Gather Context
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
    
    # Time/Date Inference Logic
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

    # Initial State
    initial_input = {
        "profile_name": profile_name,
        "year": final_year,
        "date_str": final_date,
        "time_str": final_time,
        "scene_title": title,
        "scene_brief": brief,
        "recent_context": context_str,
        "revision_count": 0,
        "critique_notes": "",
        "is_grounded": False,
        "current_draft": "",
        "banned_words": ""
    }
    
    final_state = app.invoke(initial_input)
    
    # Save to Disk
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title).replace(" ", "_")
    safe_date = final_date.replace(" ", "-")
    filename = f"{final_year}-{safe_date}_{safe_title}.txt"
    paths = get_paths(profile_name)
    filepath = os.path.join(paths['output'], filename)
    
    # Collision Avoidance
    counter = 1
    while os.path.exists(filepath):
        filename = f"{final_year}-{safe_date}_{safe_title}_{counter}.txt"
        filepath = os.path.join(paths['output'], filename)
        counter += 1
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_state['current_draft'])
        
    return final_state['current_draft'], filepath

# --- AUXILIARY AI TOOLS ---

def extract_dynamic_spoilers(plan, year, profile_name, settings=None):
    """Parses future events from the 'Plan' to prevent AI from revealing them too early."""
    if not plan or plan == "NO PLAN.": return []
    prompt = f"List FUTURE events after {year} from: {plan}. OUTPUT: Comma-separated."
    llm = get_llm(profile_name, "chat", settings=settings) 
    return [x.strip() for x in llm.invoke([HumanMessage(content=prompt)]).content.split(',')]

def infer_header_data(brief, prev_context, settings, profile_name):
    """Uses LLM to deduce the Scene's Date/Time if the user left it blank."""
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
        clean = res.replace("```json", "").replace("```", "")
        return json.loads(clean)
    except: 
        return {}

def run_chat_query(profile_name, user_input):
    """General purpose Co-Author chat that is aware of the entire RAG context."""
    lore, rules, plan, facts, spoilers = get_full_context_data(profile_name)
    state = get_world_state(profile_name)
    
    prompt = f"""
    ROLE: Co-Author. 
    CONTEXT: {facts}, {plan[:2000]}. 
    WORLD RULES: {rules}
    STATE: {state}. 
    USER: {user_input}
    """
    llm = get_llm(profile_name, "chat")
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_reaction_for_scene(profile_name, filename, faction):
    """Simulates a specific faction's reaction to a written scene."""
    content = read_file_content(profile_name, filename)
    prompt = f"ROLE: Sim. FACTION: {faction}. SCENE: {content}. React if alive."
    
    llm = get_llm(profile_name, "reaction")
    res = llm.invoke([HumanMessage(content=prompt)]).content
    if "REFUSAL" in res: return False, res
    paths = get_paths(profile_name)
    with open(os.path.join(paths['output'], filename), "a", encoding="utf-8") as f:
        f.write(f"\n\n>>> REACTION: {faction} <<<\n{res}\n")
    return True, res

def run_war_room_simulation(profile_name, action_input):
    """
    Performs a Monte Carlo-style simulation of a proposed action using 
    the current world state, assets, and rules. Returns a risk report.
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

# --- FILE HELPERS ---

def get_all_files_list(profile_name: str) -> List[str]:
    """Returns all generated scenes and lore files sorted by modification time."""
    paths = get_paths(profile_name)
    scenes = glob.glob(os.path.join(paths['output'], "*.txt"))
    lore = glob.glob(os.path.join(paths['lore'], "*.txt"))
    all_files = []
    for f in scenes: all_files.append((os.path.basename(f), f, os.path.getmtime(f)))
    for f in lore: all_files.append((os.path.basename(f), f, os.path.getmtime(f)))
    all_files.sort(key=lambda x: x[2], reverse=True)
    return [x[0] for x in all_files]

def read_file_content(profile_name, filename):
    paths = get_paths(profile_name)
    p1 = os.path.join(paths['output'], filename)
    if os.path.exists(p1): return open(p1, "r", encoding="utf-8").read()
    p2 = os.path.join(paths['lore'], filename)
    if os.path.exists(p2): return open(p2, "r", encoding="utf-8").read()
    return "Error: File not found."

def save_edited_scene(profile_name, filename, content):
    paths = get_paths(profile_name)
    try:
        with open(os.path.join(paths['output'], filename), "w", encoding="utf-8") as f: 
            f.write(content)
        return True, "Saved"
    except Exception as e: 
        return False, str(e)

def delete_specific_scene(profile_name, filename):
    paths = get_paths(profile_name)
    p = os.path.join(paths['output'], filename)
    if os.path.exists(p):
        os.remove(p)
        return True, "Deleted"
    return False, "Cannot delete Lore"

def get_last_scenes(profile_name):
    """Fetches the last 3 generated scenes for narrative continuity."""
    paths = get_paths(profile_name)
    files = glob.glob(os.path.join(paths['output'], "*.txt"))
    if not files: return "NO SCENES."
    files.sort(key=os.path.getmtime)
    context = ""
    for f in files[-3:]:
        context += f"\n=== PREV: {os.path.basename(f)} ===\n{open(f, 'r', encoding='utf-8').read()[:3000]}\n"
    return context

def compile_manuscript(profile_name, files):
    """Concatenates selected files into a single manuscript string."""
    return "\n***\n".join([read_file_content(profile_name, f) for f in files])

# --- CHAT PERSISTENCE ---

def get_chat_history(profile_name):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

def save_chat_message(profile_name, role, content):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def clear_chat_history(profile_name):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()