"""
Chronos Story Director - Data Ingestion Utility
===============================================
A command-line tool for bulk importing text and PDF documents into 
the Chronos RAG database. Useful for initial project setup or 
mass-ingesting lore libraries.

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
import glob
import sqlite3
from typing import Optional

# Third-Party Imports
from pypdf import PdfReader

# Internal Engine Import
import backend as engine

def read_text_safe(filepath: str) -> str:
    """
    Reads a text file with encoding fallback. 
    Attempts UTF-8 first, then falls back to ISO-8859-1 (Latin-1).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        print(f"  [WARN] UTF-8 decoding failed for {os.path.basename(filepath)}. Fallback to ISO-8859-1.")
        with open(filepath, "r", encoding="ISO-8859-1") as f:
            return f.read()

def ingest_profile_data(profile_name: str):
    """
    Scans the profile's 'input' directories (lore, plans, rules) and 
    inserts new files into the SQLite database.
    """
    paths = engine.get_paths(profile_name)
    db_path = paths['db']
    
    # Define Source Directories
    # Structure: profiles/{name}/input/{type}
    base_input = os.path.dirname(paths['lore'])
    source_dirs = {
        "Lore": paths['lore'],
        "Plan": os.path.join(base_input, "plans"),
        "Rulebook": os.path.join(base_input, "rules")
    }

    # Auto-create directories if missing
    for d in source_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"\n--- 📂 INGESTING TARGET: {profile_name} ---")
    print(f"Database: {db_path}")

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        c = conn.cursor()
    except sqlite3.Error as e:
        print(f"[ERROR] Could not connect to database: {e}")
        return

    # Processing Loop
    for doc_type, folder_path in source_dirs.items():
        if not os.path.exists(folder_path): 
            continue
        
        # Collect supported files
        files = glob.glob(os.path.join(folder_path, "*.txt")) + \
                glob.glob(os.path.join(folder_path, "*.pdf"))
        
        if files:
            print(f"\nScanning [{doc_type}] in: {folder_path}...")
        
        new_count = 0
        
        for filepath in files:
            filename = os.path.basename(filepath)
            
            # Idempotency Check (Skip if already exists)
            c.execute("SELECT id FROM memory_fragments WHERE source_filename = ?", (filename,))
            if c.fetchone():
                continue

            content: Optional[str] = None
            
            # Handler: Text Files
            if filename.endswith(".txt"):
                content = read_text_safe(filepath)
            
            # Handler: PDF Files
            elif filename.endswith(".pdf"):
                try:
                    reader = PdfReader(filepath)
                    content = "".join([page.extract_text() + "\n" for page in reader.pages])
                except Exception as e:
                    print(f"  [ERROR] PDF Parsing Failed for {filename}: {e}")
                    continue
            
            # Insert into DB
            if content:
                print(f"  [+] Importing: {filename}")
                c.execute(
                    "INSERT INTO memory_fragments (source_filename, content, type) VALUES (?, ?, ?)", 
                    (filename, content, doc_type)
                )
                new_count += 1

        if new_count == 0 and files:
            print("  (No new files found)")

    conn.commit()
    conn.close()
    print("\n--- ✅ Ingestion Complete ---")

def main():
    print("========================================")
    print("   CHRONOS DATA INGESTION UTILITY")
    print("========================================")
    
    profiles = engine.list_profiles()
    
    if not profiles:
        print("\n[!] No profiles detected.")
        print("Please launch the main application to create a profile first.")
        return

    print("\nAvailable Profiles:")
    for i, p in enumerate(profiles):
        print(f"{i+1}. {p}")
    
    choice = input("\nSelect Profile Number (or 'q' to quit): ")
    if choice.lower() in ['q', 'quit', 'exit']:
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(profiles):
            target_profile = profiles[idx]
            ingest_profile_data(target_profile)
        else:
            print("[!] Invalid selection index.")
    except ValueError:
        print("[!] Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()