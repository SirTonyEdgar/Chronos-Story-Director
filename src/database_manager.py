"""
Chronos Story Director - Database & Persistence Layer
=====================================================
The centralized "Source of Truth". 
Handles all SQLite connections, File I/O, and State Management.
"""

import os
import re
import glob
import json
import sqlite3
import shutil
import datetime
from typing import List, Dict, Optional, Any
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")

# ==========================================
# 1. FILE SYSTEM & PATHS
# ==========================================

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

def list_profiles() -> List[str]:
    if not os.path.exists(PROFILES_DIR): return []
    return [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d))]

def ensure_profile_structure(profile_name: str):
    paths = get_paths(profile_name)
    for p in [paths['root'], paths['data'], paths['output'], paths['lore']]:
        os.makedirs(p, exist_ok=True)
    
    init_db(profile_name)
    
    if not os.path.exists(paths['state']):
        default_state = {"Status": "New Game", "Assets": [], "Allies": [], "Projects": []}
        with open(paths['state'], 'w') as f: 
            json.dump(default_state, f, indent=4)
    return paths

def get_all_files_list(profile_name: str) -> List[str]:
    """Returns a sorted list of SCENE files only (newest first)."""
    paths = get_paths(profile_name)
    scenes = glob.glob(os.path.join(paths['output'], "*.txt"))
    
    all_files = []
    for f in scenes: 
        all_files.append((os.path.basename(f), f, os.path.getmtime(f)))
    
    all_files.sort(key=lambda x: x[2], reverse=True)
    return [x[0] for x in all_files]

def read_text_safe(filepath: str) -> str:
    """Robust text reader that handles encoding errors (UTF-8 vs CP1252)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="cp1252") as f:
                return f.read()
        except:
            return "[Error: Unknown File Encoding]"

def read_file_content(profile_name, filename):
    """Safely reads the content of a file from either Output or Lore directories."""
    paths = get_paths(profile_name)
    p1 = os.path.join(paths['output'], filename)
    if os.path.exists(p1): return read_text_safe(p1)
    p2 = os.path.join(paths['lore'], filename)
    if os.path.exists(p2): return read_text_safe(p2)
    return "Error: File not found."

def get_fragment_path(profile_name: str, doc_type: str, filename: str) -> Optional[str]:
    paths = get_paths(profile_name)
    base_input = os.path.dirname(paths['lore'])
    
    type_map = {
        "Lore": "lore", "Plan": "plans", "Rulebook": "rules", "Fact": "facts"
    }
    
    if doc_type not in type_map: return None

    folder_name = type_map.get(doc_type, "lore")
    folder_path = os.path.join(base_input, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    safe_name = re.sub(r'[\\/*?:"<>|]', "", filename).replace(" ", "_")
    if not safe_name.endswith(".txt"): safe_name += ".txt"
        
    return os.path.join(folder_path, safe_name)

# ==========================================
# 2. DATABASE OPERATIONS (SQLite)
# ==========================================

def init_db(profile_name: str):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30) 
    c = conn.cursor()
    
    # 1. Create the table with metadata and timeline natively
    c.execute('''CREATE TABLE IF NOT EXISTS memory_fragments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_filename TEXT, 
        content TEXT, 
        type TEXT, 
        year INTEGER DEFAULT NULL,
        metadata TEXT DEFAULT '',
        timeline TEXT DEFAULT ''
    )''')
    
    # Add metadata column to older databases
    try:
        c.execute("ALTER TABLE memory_fragments ADD COLUMN metadata TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass 
        
    # Add timeline column to older databases
    try:
        c.execute("ALTER TABLE memory_fragments ADD COLUMN timeline TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass

    c.execute('''CREATE TABLE IF NOT EXISTS story_settings (
        key TEXT PRIMARY KEY, value TEXT
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS faction_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        faction_name TEXT, reaction_text TEXT, source_scene TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()
    conn.close()

# ==========================================
# --- SETTINGS ---
# ==========================================
def update_story_setting(profile_name: str, key: str, value: str):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO story_settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

def get_story_settings(profile_name: str) -> dict:
    defaults = {
        "protagonist": "The Protagonist",
        "default_timezone": "UTC",
        "use_time_system": "true",
        "enable_chapters": "true",
        "model_scene": "gemini-2.5-pro",
        "model_chat": "gemini-2.5-flash",
    }
    paths = get_paths(profile_name)
    try:
        conn = sqlite3.connect(paths['db'])
        c = conn.cursor()
        c.execute("SELECT key, value FROM story_settings")
        for k, v in c.fetchall():
            defaults[k] = v
        conn.close()
    except sqlite3.Error: pass 
    return defaults

# ==========================================
# --- KNOWLEDGE BASE (CRUD) ---
# ==========================================
def add_fragment(profile_name, filename, content, doc_type, timeline=""):
    """Persists a new document to DB and File System, tagged with an optional timeline."""
    init_db(profile_name)
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    c.execute(
        "INSERT INTO memory_fragments (source_filename, content, type, timeline) VALUES (?, ?, ?, ?)", 
        (filename, content, doc_type, timeline)
    )
    conn.commit()
    conn.close()

    file_path = get_fragment_path(profile_name, doc_type, filename)
    if file_path:
        try:
            with open(file_path, "w", encoding="utf-8") as f: f.write(content)
        except Exception as e: print(f"File Mirror Error: {e}")

def get_fragments(profile_name: str, doc_type: Optional[str] = None):
    """Retrieves fragments, now including the timeline data."""
    init_db(profile_name)
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    if doc_type: 
        c.execute("SELECT id, source_filename, content, type, metadata, timeline FROM memory_fragments WHERE type = ? ORDER BY id DESC", (doc_type,))
    else: 
        c.execute("SELECT id, source_filename, content, type, metadata, timeline FROM memory_fragments ORDER BY type, id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def get_content_by_ids(profile_name: str, id_list: List[int]) -> str:
    """Efficiently retrieves content for a list of IDs in a single query."""
    if not id_list: return ""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    
    placeholders = ','.join(['?'] * len(id_list))
    query = f"SELECT content FROM memory_fragments WHERE id IN ({placeholders})"
    
    try:
        c.execute(query, id_list)
        rows = c.fetchall()
        return "\n\n".join([r[0] for r in rows])
    except Exception as e:
        print(f"Batch Fetch Error: {e}")
        return ""
    finally:
        conn.close()

def update_fragment(profile_name, frag_id, new_content, new_timeline=None):
    """Updates content and optionally changes the assigned timeline."""
    init_db(profile_name)
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    
    c.execute("SELECT source_filename, type, timeline FROM memory_fragments WHERE id = ?", (frag_id,))
    row = c.fetchone()
    
    if row:
        filename, doc_type, current_timeline = row

        final_timeline = new_timeline if new_timeline is not None else current_timeline
        
        c.execute("UPDATE memory_fragments SET content = ?, timeline = ? WHERE id = ?", (new_content, final_timeline, frag_id))
        conn.commit()
        
        file_path = get_fragment_path(profile_name, doc_type, filename)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f: f.write(new_content)
            
    conn.close()

def delete_fragment(profile_name, frag_id):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    
    c.execute("SELECT source_filename, type FROM memory_fragments WHERE id = ?", (frag_id,))
    row = c.fetchone()
    
    if row:
        filename, doc_type = row
        c.execute("DELETE FROM memory_fragments WHERE id = ?", (frag_id,))
        conn.commit()
        
        file_path = get_fragment_path(profile_name, doc_type, filename)
        if file_path and os.path.exists(file_path): os.remove(file_path)
            
    conn.close()

def rename_fragment(profile_name, frag_id, new_filename):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("SELECT source_filename, type FROM memory_fragments WHERE id = ?", (frag_id,))
    row = c.fetchone()
    
    if row:
        old_filename, doc_type = row
        c.execute("UPDATE memory_fragments SET source_filename = ? WHERE id = ?", (new_filename, frag_id))
        conn.commit()
        
        old_path = get_fragment_path(profile_name, doc_type, old_filename)
        new_path = get_fragment_path(profile_name, doc_type, new_filename)
        if old_path and new_path and os.path.exists(old_path): os.rename(old_path, new_path)

    conn.close()

def upsert_scene(profile_name: str, filename: str, content: str, metadata: str = ""):
    """
    Inserts or Updates a Scene entry in the database, including its AI summary metadata.
    """
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    
    c.execute("SELECT id FROM memory_fragments WHERE source_filename = ? AND type = 'Scene'", (filename,))
    row = c.fetchone()
    
    if row:
        # UPDATE existing scene
        c.execute("UPDATE memory_fragments SET content = ?, metadata = ? WHERE id = ?", (content, metadata, row[0]))
    else:
        # INSERT new scene
        c.execute("INSERT INTO memory_fragments (source_filename, content, type, metadata) VALUES (?, ?, 'Scene', ?)", (filename, content, metadata))
        
    conn.commit()
    conn.close()

def archive_scene_db(profile_name: str, filename: str):
    """
    Soft deletes a scene from the active context by changing its type to 'Archived_Scene'.
    Used when merging chapters so the AI doesn't see double content (Part 1 vs Merged Ch 1).
    """
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("UPDATE memory_fragments SET type = 'Archived_Scene' WHERE source_filename = ? AND type = 'Scene'", (filename,))
    conn.commit()
    conn.close()

def delete_scene_db(profile_name: str, filename: str):
    """
    Hard deletes a scene entry from the database.
    Used when the user explicitly deletes a file.
    """
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("DELETE FROM memory_fragments WHERE source_filename = ? AND type = 'Scene'", (filename,))
    conn.commit()
    conn.close()

# ==========================================
# --- WORLD STATE ---
# ==========================================
def get_world_state(profile_name: str) -> Dict:
    paths = get_paths(profile_name)
    try:
        with open(paths['state'], 'r') as f: return json.load(f)
    except: return {}

def save_world_state(profile_name: str, new_state_dict: Dict):
    paths = get_paths(profile_name)
    try:
        # Auto-Backup
        if os.path.exists(paths['state']):
            backup_name = f"world_state_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy(paths['state'], os.path.join(paths['data'], backup_name))
            
            backups = sorted(glob.glob(os.path.join(paths['data'], "world_state_backup_*.json")))
            for old_b in backups[:-5]: os.remove(old_b)

        with open(paths['state'], 'w') as f: json.dump(new_state_dict, f, indent=4)
        return True, "State Saved"
    except Exception as e: return False, str(e)

# ==========================================
# --- PROJECT MANAGEMENT (WORLD STATE) ---
# ==========================================

def add_project(profile_name: str, name: str, description: str, faction: str):
    """Adds a new project to the World State."""
    state = get_world_state(profile_name)
    if "Projects" not in state:
        state["Projects"] = []
        
    new_project = {
        "Name": name,
        "Description": description,
        "Faction": faction,
        "Progress": 0,
        "Notes": ""
    }
    state["Projects"].append(new_project)
    return save_world_state(profile_name, state)

def update_project(profile_name: str, index: int, progress: int, notes: str, new_name=None, new_desc=None):
    """Updates an existing project's progress and notes."""
    state = get_world_state(profile_name)
    if "Projects" not in state or index < 0 or index >= len(state["Projects"]):
        return False, "Project not found."
        
    proj = state["Projects"][index]
    proj["Progress"] = progress
    proj["Notes"] = notes
    if new_name: proj["Name"] = new_name
    if new_desc: proj["Description"] = new_desc
    
    return save_world_state(profile_name, state)

def complete_project(profile_name: str, index: int, lore_summary: str, doc_type="Fact"):
    """
    Removes a completed project from the World State AND automatically 
    saves its completion as a historical Fact/Lore in the database.
    """
    state = get_world_state(profile_name)
    if "Projects" not in state or index < 0 or index >= len(state["Projects"]):
        return False, "Project not found."
        
    # Remove from active state
    proj = state["Projects"].pop(index)
    save_world_state(profile_name, state)
    
    # Convert the completed project into permanent historical memory
    safe_name = re.sub(r'[\\/*?:"<>|]', "", proj['Name']).replace(" ", "_")
    filename = f"Project_Completed_{safe_name}.txt"
    
    # Save to the Knowledge Base so the RAG remembers it happened
    add_fragment(profile_name, filename, lore_summary, doc_type)
    
    return True, f"Project '{proj['Name']}' completed and archived to Knowledge Base."

# ==========================================
# --- CHAT & REACTION HISTORY ---
# ==========================================
def get_chat_history(profile_name):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history ORDER BY id ASC")
    rows = [{"role": r[0], "content": r[1]} for r in c.fetchall()]
    conn.close()
    return rows

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

def get_all_faction_memories(profile_name: str) -> list:
    """Retrieves all faction reactions for the frontend to display in a list/table."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    try:
        c.execute("SELECT id, faction_name, reaction_text, source_scene, timestamp FROM faction_memory ORDER BY id DESC")
        rows = [{"id": r[0], "faction": r[1], "text": r[2], "scene": r[3], "timestamp": r[4]} for r in c.fetchall()]
        return rows
    except Exception as e:
        print(f"Error fetching faction memories: {e}")
        return []
    finally:
        conn.close()

def update_faction_reaction(profile_name: str, reaction_id: int, new_text: str, new_faction: str):
    """Updates the text of a specific faction memory."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("UPDATE faction_memory SET reaction_text = ?, faction_name = ? WHERE id = ?", (new_text, new_faction, reaction_id))
    conn.commit()
    conn.close()

def delete_faction_reaction(profile_name: str, reaction_id: int):
    """Deletes a specific faction memory entirely."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("DELETE FROM faction_memory WHERE id = ?", (reaction_id,))
    conn.commit()
    conn.close()

def save_faction_reaction(profile_name, faction, text, scene_name):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'], timeout=30)
    c = conn.cursor()
    c.execute("INSERT INTO faction_memory (faction_name, reaction_text, source_scene) VALUES (?, ?, ?)", 
              (faction, text, scene_name))
    conn.commit()
    conn.close()

def get_recent_faction_memory(profile_name, faction, limit=3):
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    try:
        c.execute("SELECT reaction_text, source_scene FROM faction_memory WHERE faction_name = ? ORDER BY id DESC LIMIT ?", (faction, limit))
        rows = c.fetchall()
    except: return ""
    conn.close()
    
    if not rows: return "No previous records found."
    return "\n".join([f"--- FROM SCENE: {r[1]} ---\n{r[0][:800]}...\n" for r in rows])

def get_distinct_factions(profile_name: str) -> List[str]:
    """Retrieves list of all unique factions ever simulated."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    try:
        c.execute("SELECT DISTINCT faction_name FROM faction_memory")
        return [r[0] for r in c.fetchall()]
    except: return []
    finally: conn.close()

def delete_last_faction_reaction(profile_name, faction):
    """Removes the most recent memory entry for a specific faction."""
    paths = get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("DELETE FROM faction_memory WHERE id = (SELECT MAX(id) FROM faction_memory WHERE faction_name = ?)", (faction,))
    conn.commit()
    conn.close()

# ==========================================
# 3. CLI UTILITIES (Legacy Support)
# ==========================================
def ingest_profile_data(profile_name: str):
    """CLI Tool: Imports text files into DB."""

    init_db(profile_name)

    paths = get_paths(profile_name)
    db_path = paths['db']
    base_input = os.path.dirname(paths['lore'])
    source_dirs = {
        "Lore": paths['lore'],
        "Plan": os.path.join(base_input, "plans"),
        "Rulebook": os.path.join(base_input, "rules"),
        "Fact": os.path.join(base_input, "facts")
    }

    print(f"\n--- 📂 INGESTING: {profile_name} ---")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    for doc_type, folder_path in source_dirs.items():
        if not os.path.exists(folder_path): continue
        files = glob.glob(os.path.join(folder_path, "*.txt")) + glob.glob(os.path.join(folder_path, "*.pdf"))
        
        for filepath in files:
            filename = os.path.basename(filepath)
            c.execute("SELECT id FROM memory_fragments WHERE source_filename = ? AND type = ?", (filename, doc_type))
            if c.fetchone(): continue

            content = None
            if filename.endswith(".txt"): content = read_text_safe(filepath)
            elif filename.endswith(".pdf") and PdfReader:
                try: content = "".join([page.extract_text() for page in PdfReader(filepath).pages])
                except: pass
            
            if content:
                print(f"  [+] Importing: {filename}")
                c.execute("INSERT INTO memory_fragments (source_filename, content, type) VALUES (?, ?, ?)", (filename, content, doc_type))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    import sys
    
    # If the user runs: python database_manager.py "Specific_Profile_Name"
    if len(sys.argv) > 1:
        target_profile = sys.argv[1]
        print(f"Targeting specific profile: {target_profile}")
        ingest_profile_data(target_profile)
    else:
        # If the user just runs: python database_manager.py
        all_profiles = list_profiles()
        
        if not all_profiles:
            print("No profiles found in the 'profiles' directory.")
        else:
            print(f"Found {len(all_profiles)} profiles. Beginning batch ingestion...")
            for profile in all_profiles:
                ingest_profile_data(profile)
            
            print("\n✅ All profiles successfully ingested and synced to their databases!")