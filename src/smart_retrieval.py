"""
Chronos Smart Retrieval Engine (src/smart_retrieval.py)
=======================================================
Handles "Librarian" logic: selecting relevant Lore/Facts/History 
before full content ingestion to save tokens and improve focus.

Copyright (c) 2025 SirTonyEdgar
Licensed under the MIT License.
"""

import json
import sqlite3
from langchain_core.messages import HumanMessage
import backend as engine  # Import core tools

def get_relevant_fragment_ids(profile_name, user_query, doc_types=None):
    """
    Scans the 'Table of Contents' (Titles) of the database and asks the AI 
    which entries are relevant to the user's query.
    
    Args:
        doc_types: List of types to search (e.g. ['Lore', 'Fact']). If None, searches all.
    Returns:
        List of integer IDs.
    """
    paths = engine.get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    
    # Fetch Titles Only
    if doc_types:
        placeholders = ','.join(['?'] * len(doc_types))
        query = f"SELECT id, source_filename, type FROM memory_fragments WHERE type IN ({placeholders})"
        c.execute(query, doc_types)
    else:
        c.execute("SELECT id, source_filename, type FROM memory_fragments")
        
    rows = c.fetchall()
    conn.close()
    
    if not rows: return []

    # Format the "Menu" for the AI
    # Format: "ID: 1 | Title: The Betrayal of 1999 (Lore)"
    toc_list = []
    for r in rows:
        toc_list.append(f"ID: {r[0]} | Title: {r[1]} ({r[2]})")
    
    toc_str = "\n".join(toc_list)

    # The "Librarian" Prompt
    prompt = f"""
    ROLE: Database Librarian.
    TASK: Select relevant document IDs based on the user's need.
    
    *** AVAILABLE DOCUMENTS ***
    {toc_str}
    
    *** USER SCENARIO / QUERY ***
    "{user_query}"
    
    *** INSTRUCTION ***
    Analyze the scenario. Identify which documents (if any) might contain necessary background info.
    - If the user mentions a specific location, faction, or event, select its document.
    - Select ONLY highly relevant items.
    - Max 5 items.
    
    OUTPUT FORMAT: JSON List of integers ONLY. Example: [1, 14, 22]
    If nothing is relevant, output: []
    """
    
    # AI Execution
    llm = engine.get_llm(profile_name, "retrieval") 
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content

        ids = engine._extract_json(res) 
        
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
    paths = engine.get_paths(profile_name)
    conn = sqlite3.connect(paths['db'])
    c = conn.cursor()
    c.execute("SELECT DISTINCT faction_name FROM faction_memory")
    rows = c.fetchall()
    conn.close()
    
    existing_factions = [r[0] for r in rows]
    if not existing_factions: return user_input

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
    
    llm = engine.get_llm(profile_name, "retrieval")
    res = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    res = res.replace('"', '').replace("'", "")
    
    if res == "NEW" or res not in existing_factions:
        return user_input
    return res