"""
Chronos Story Director - Data Ingestion Utility
===============================================
A command-line tool for bulk importing text and PDF documents into 
the Chronos RAG database.

Copyright (c) 2025 SirTonyEdgar
"""

import os
import re
import glob
import sqlite3
from typing import Optional

# Third-Party Imports
from pypdf import PdfReader

# Internal Engine Import
import backend as engine

def read_text_safe(filepath: str) -> str:
    """Reads a text file with encoding fallback."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        print(f"  [WARN] UTF-8 decoding failed for {os.path.basename(filepath)}. Fallback to ISO-8859-1.")
        with open(filepath, "r", encoding="ISO-8859-1") as f:
            return f.read()

def ingest_profile_data(profile_name: str):
    """Scans the profile's 'input' directories and inserts new files."""
    paths = engine.get_paths(profile_name)
    db_path = paths['db']
    
    base_input = os.path.dirname(paths['lore'])
    source_dirs = {
        "Lore": paths['lore'],
        "Plan": os.path.join(base_input, "plans"),
        "Rulebook": os.path.join(base_input, "rules")
    }

    # Auto-create directories
    for d in source_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"\n--- 📂 INGESTING LORE & RULES: {profile_name} ---")

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        c = conn.cursor()
    except sqlite3.Error as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    for doc_type, folder_path in source_dirs.items():
        if not os.path.exists(folder_path): continue
        
        files = glob.glob(os.path.join(folder_path, "*.txt")) + \
                glob.glob(os.path.join(folder_path, "*.pdf"))
        
        if files: print(f"\nScanning [{doc_type}] in: {folder_path}...")
        
        new_count = 0
        for filepath in files:
            filename = os.path.basename(filepath)
            
            # Idempotency Check
            c.execute("SELECT id FROM memory_fragments WHERE source_filename = ?", (filename,))
            if c.fetchone(): continue

            content = None
            if filename.endswith(".txt"):
                content = read_text_safe(filepath)
            elif filename.endswith(".pdf"):
                try:
                    reader = PdfReader(filepath)
                    content = "".join([page.extract_text() + "\n" for page in reader.pages])
                except Exception as e:
                    print(f"  [ERROR] PDF Parsing Failed for {filename}: {e}")
                    continue
            
            if content:
                print(f"  [+] Importing: {filename}")
                c.execute("INSERT INTO memory_fragments (source_filename, content, type) VALUES (?, ?, ?)", 
                          (filename, content, doc_type))
                new_count += 1

        if new_count == 0 and files: print("  (No new files found)")

    conn.commit()
    conn.close()
    print("\n--- ✅ Ingestion Complete ---")

def backfill_reactions(profile_name: str):
    """
    Scans existing scene files in 'output/scenes', finds '>>> REACTION' blocks,
    and populates the 'faction_memory' table so the AI remembers past voices.
    """
    paths = engine.get_paths(profile_name)
    db_path = paths['db']
    scene_dir = paths['output']
    
    print(f"\n--- 🗣️ BACKFILLING REACTION MEMORY: {profile_name} ---")
    
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS faction_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faction_name TEXT,
            reaction_text TEXT,
            source_scene TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
    except sqlite3.Error as e:
        print(f"[ERROR] Database error: {e}")
        return

    files = glob.glob(os.path.join(scene_dir, "*.txt"))
    files.sort()
    
    count = 0
    
    for filepath in files:
        filename = os.path.basename(filepath)
        content = read_text_safe(filepath)
        
        if ">>> REACTION:" in content:
            parts = re.split(r'>>> REACTION:', content)
            
            for part in parts[1:]:
                header_match = re.search(r'\s*(.*?) \|\s*(.*?) <<<', part)
                if header_match:
                    faction = header_match.group(1).strip()
                    
                    body = re.sub(r'\s*(.*?) \|\s*(.*?) <<<(\n✨.*)?', '', part, count=1).strip()

                    c.execute("SELECT id FROM faction_memory WHERE source_scene = ? AND faction_name = ?", (filename, faction))
                    if not c.fetchone():
                        c.execute("INSERT INTO faction_memory (faction_name, reaction_text, source_scene) VALUES (?, ?, ?)",
                                  (faction, body, filename))
                        print(f"  [+] Learned Voice: {faction} (from {filename})")
                        count += 1
    
    conn.commit()
    conn.close()
    print(f"\n--- ✅ Backfill Complete. Imported {count} legacy reactions. ---")

def main():
    while True:
        print("\n========================================")
        print("   CHRONOS DATA MANAGER (v12.8)")
        print("========================================")
        
        profiles = engine.list_profiles()
        if not profiles:
            print("\n[!] No profiles detected.")
            return

        print("Available Profiles:")
        for i, p in enumerate(profiles):
            print(f"{i+1}. {p}")
        
        p_choice = input("\nSelect Profile Number (or 'q' to quit): ")
        if p_choice.lower() in ['q', 'quit', 'exit']: break
        
        try:
            idx = int(p_choice) - 1
            if not (0 <= idx < len(profiles)):
                print("[!] Invalid profile.")
                continue
            
            target_profile = profiles[idx]
            
            print(f"\nSelected: {target_profile}")
            print("1. Ingest Lore/Rules/Plans (Standard)")
            print("2. Backfill Reaction Memory (Import from Scenes)")
            
            op = input("Select Operation [1-2]: ")
            
            if op == "1":
                ingest_profile_data(target_profile)
            elif op == "2":
                backfill_reactions(target_profile)
            else:
                print("[!] Invalid operation.")
                
        except ValueError:
            print("[!] Invalid input.")

if __name__ == "__main__":
    main()