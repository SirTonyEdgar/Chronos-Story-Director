"""
Chronos Story Engine - REST API
===============================
Exposes the core narrative logic of backend.py to the React frontend.
Organized by functional modules matching the application sidebar.

Author: SirTonyEdgar
License: MIT
"""

from . import backend as engine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import shutil
import os

app = FastAPI(
    title="Chronos Story Engine API", 
    description="Backend interface for the Chronos AI-assisted storytelling platform.",
    version="2.0.0"
)

# --- CONFIGURATION: CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 📦 DATA TRANSFER OBJECTS (DTOs)
# ==========================================

# -- Scene Creator & File Management --
class SceneGenerationRequest(BaseModel):
    chapter: Optional[int] = None
    title: str = ""
    year: int = 0
    date_str: str = ""
    time_str: str = ""
    brief: str
    context_files: List[str] = []
    fog_of_war: bool = False
    timeline: str = ""

class SceneEditRequest(BaseModel):
    filename: str
    content: str

class FileListRequest(BaseModel):
    filenames: List[str]

# -- Chat --
class ChatQueryRequest(BaseModel):
    prompt: str
    timeline: str = ""

# -- War Room --
class SimulationRequest(BaseModel):
    scenario: str
    timeline: str = ""

# -- Knowledge Base --
class KnowledgeItem(BaseModel):
    name: str
    content: str
    category: str
    id: Optional[int] = None
    timeline: str = ""

class DeleteRequest(BaseModel):
    id: int

# -- World State --
class AnalysisRequest(BaseModel):
    filenames: List[str]
    timeline: str = ""

class AssetsPayload(BaseModel):
    assets: List[Dict[str, Any]]

# -- Project Endpoints --

class ProjectRequest(BaseModel):
    name: str
    description: str
    faction: str

class ProjectUpdateRequest(BaseModel):
    progress: int
    notes: str
    new_name: Optional[str] = None
    new_desc: Optional[str] = None

class ProjectCompleteRequest(BaseModel):
    lore_summary: str

# -- Network Map --
class GraphUpdate(BaseModel):
    updates: List[Dict[str, Any]]

# -- Reaction Tool --
class ReactionRequest(BaseModel):
    scene_file: str
    faction: str
    format_style: str
    public_only: bool
    custom_instructions: str = ""
    timeline: str = ""

class EditReactionRequest(BaseModel):
    new_text: str
    new_faction: str

class UndoReactionRequest(BaseModel):
    scene_file: str
    faction: str

# -- Compiler --
class CompileRequest(BaseModel):
    filenames: List[str]


# ==========================================
# 0. 👤 PROFILE MANAGEMENT
# ==========================================

@app.get("/profiles/list")
def list_available_profiles():
    """Returns a list of all available profile folders."""
    return engine.list_profiles()

@app.post("/profiles/create")
def create_new_profile(name: str):
    """Creates a new profile directory structure."""
    try:
        # Sanitize name to be folder-safe
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip().replace(" ", "_")
        if not clean_name:
            raise HTTPException(status_code=400, detail="Invalid profile name")
            
        engine.ensure_profile_structure(clean_name)
        return {"status": "Created", "name": clean_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profiles/export/{profile}")
def export_profile_zip(profile: str):
    """Zips the entire profile folder and returns it as a download."""
    try:
        paths = engine.get_paths(profile)
        # Create a zip file in the data directory to avoid cluttering root
        zip_base = os.path.join(paths['data'], f"{profile}_Backup")
        
        # Create the archive
        zip_path = shutil.make_archive(zip_base, 'zip', paths['root'])
        
        return FileResponse(
            zip_path, 
            filename=f"{profile}_Backup.zip", 
            media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 1. 🎬 SCENE CREATOR MODULE
# ==========================================

@app.get("/files/{profile}")
def list_project_files(profile: str):
    """Retrieves a list of all scene files in the project."""
    return engine.get_all_files_list(profile)

@app.get("/file/{profile}/{filename}")
def read_scene_content(profile: str, filename: str):
    """Reads the raw text content of a specific scene file."""
    content = engine.read_file_content(profile, filename)
    return {"content": content}

@app.get("/next_chapter/{profile}")
def predict_next_chapter(profile: str):
    """Calculates the next available chapter number based on existing files."""
    return {"next_chapter": engine.get_next_chapter_number(profile)}

@app.post("/scene/generate/{profile}")
def generate_new_scene(profile: str, payload: SceneGenerationRequest):
    """Triggers the AI to write a new scene based on the brief and context."""
    try:
        text, path = engine.generate_scene(
            profile, payload.chapter, payload.year, payload.date_str, 
            payload.time_str, payload.title, payload.brief, 
            payload.context_files, use_fog_of_war=payload.fog_of_war,
            timeline=payload.timeline
        )
        return {"status": "Success", "filename": os.path.basename(path), "content": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scene/save/{profile}")
def save_scene_edits(profile: str, payload: SceneEditRequest):
    """Overwrites a scene file with manual edits."""
    success, msg = engine.save_edited_scene(profile, payload.filename, payload.content)
    if not success: 
        raise HTTPException(status_code=500, detail=msg)
    return {"status": "Saved"}

@app.delete("/scene/{profile}/{filename}")
def delete_scene_file(profile: str, filename: str):
    """Permanently deletes a scene file."""
    count = engine.bulk_delete_files(profile, [filename])
    if count == 0: 
        raise HTTPException(status_code=400, detail="File not found or could not be deleted.")
    return {"status": "Deleted"}

@app.post("/merge/scenes/{profile}")
def merge_selected_scenes(profile: str, payload: FileListRequest):
    """
    Stitches selected chapter parts (e.g., 'Chapter 1 Part 1', 'Chapter 1 Part 2') 
    into a single merged Scene file.
    """
    try:
        if len(payload.filenames) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 files to merge.")
            
        new_filename = engine.merge_specific_files(profile, payload.filenames)
        return {"filename": new_filename, "status": "Merged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/files/bulk_delete/{profile}")
def bulk_delete_files(profile: str, payload: FileListRequest):
    """
    Permanently deletes multiple selected files at once.
    """
    try:
        count = engine.bulk_delete_files(profile, payload.filenames)
        return {"deleted_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 2. 🧠 CO-AUTHOR CHAT MODULE
# ==========================================

@app.get("/chat/history/{profile}")
def get_chat_history(profile: str):
    """Retrieves the persistent chat history for the project."""
    return engine.get_chat_history(profile)

@app.post("/chat/query/{profile}")
def query_co_author(profile: str, payload: ChatQueryRequest):
    """Sends a user prompt to the Co-Author AI and retrieves the response."""
    try:
        # Save User Message
        engine.save_chat_message(profile, "user", payload.prompt)
        # Generate Response
        response_text = engine.run_chat_query(profile, payload.prompt, timeline=payload.timeline)
        # Save Assistant Message
        engine.save_chat_message(profile, "assistant", response_text)
        
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/clear/{profile}")
def clear_chat_history(profile: str):
    """Wipes the chat history for the current profile."""
    engine.clear_chat_history(profile)
    return {"status": "History Cleared"}


# ==========================================
# 3. ⚔️ WAR ROOM MODULE
# ==========================================

@app.post("/simulation/run/{profile}")
def run_strategic_simulation(profile: str, payload: SimulationRequest):
    """Runs the Monte Carlo Strategy Simulator."""
    try:
        report = engine.run_war_room_simulation(profile, payload.scenario, timeline=payload.timeline)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 4. 🗄️ KNOWLEDGE BASE MODULE
# ==========================================

@app.get("/knowledge/list/{profile}/{category}")
def list_knowledge_fragments(profile: str, category: str):
    """Retrieves all knowledge fragments (Lore, Rules, Plans) of a specific category."""
    fragments = engine.get_fragments(profile, category)
    # Map backend tuple structure to JSON, specifically grabbing index 5 for timeline
    return [
        {
            "id": f[0], 
            "name": f[1], 
            "content": f[2], 
            "timeline": f[5] if len(f) > 5 and f[5] else ""
        } 
        for f in fragments
    ]

@app.post("/knowledge/create/{profile}")
def create_knowledge_entry(profile: str, item: KnowledgeItem):
    """Adds a new entry to the knowledge base."""
    engine.add_fragment(profile, item.name, item.content, item.category, item.timeline)
    return {"status": "Created"}

@app.post("/knowledge/update/{profile}")
def update_knowledge_entry(profile: str, item: KnowledgeItem):
    """Updates the content and title of an existing entry."""
    if not item.id:
        raise HTTPException(status_code=400, detail="ID required for update")
    
    engine.update_fragment(profile, item.id, item.content, item.timeline)
    engine.rename_fragment(profile, item.id, item.name)
    return {"status": "Updated"}

@app.post("/knowledge/delete/{profile}")
def delete_knowledge_entry(profile: str, req: DeleteRequest):
    """Removes an entry from the knowledge base."""
    engine.delete_fragment(profile, req.id)
    return {"status": "Deleted"}


# ==========================================
# 5. 📊 WORLD STATE TRACKER MODULE
# ==========================================

@app.get("/state/{profile}")
def get_world_state(profile: str):
    """Retrieves the full JSON object representing the current world state."""
    return engine.get_world_state(profile)

@app.post("/state/save/{profile}")
def save_world_state(profile: str, payload: Dict[str, Any]):
    """Overwrites the world state JSON with new data."""
    engine.save_world_state(profile, payload)
    return {"status": "State Saved"}

@app.post("/state/analyze/{profile}")
def analyze_state_changes(profile: str, payload: AnalysisRequest):
    """
    Reads selected scene files and uses AI to extract new facts (Allies, Assets, Dates).
    Returns the hypothesized new state structure.
    """
    try:
        combined_text = ""
        for fname in payload.filenames:
            content = engine.read_file_content(profile, fname)
            combined_text += f"\n=== {fname} ===\n{content}\n"

        new_state = engine.analyze_state_changes(profile, combined_text, timeline=payload.timeline)
        return new_state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assets/{profile}")
def save_specific_assets(profile: str, payload: AssetsPayload):
    """Updates only the 'Assets' list in the world state."""
    state = engine.get_world_state(profile)
    state["Assets"] = payload.assets
    engine.save_world_state(profile, state)
    return {"status": "Assets Saved"}

@app.post("/projects/create/{profile}")
def create_new_project(profile: str, payload: ProjectRequest):
    success, msg = engine.add_project(profile, payload.name, payload.description, payload.faction)
    if not success: raise HTTPException(status_code=400, detail=msg)
    return {"status": "Project Created"}

@app.post("/projects/update/{profile}/{index}")
def update_existing_project(profile: str, index: int, payload: ProjectUpdateRequest):
    success, msg = engine.update_project(profile, index, payload.progress, payload.notes, payload.new_name, payload.new_desc)
    if not success: raise HTTPException(status_code=400, detail=msg)
    return {"status": "Project Updated"}

@app.post("/projects/complete/{profile}/{index}")
def complete_and_archive_project(profile: str, index: int, payload: ProjectCompleteRequest):
    success, msg = engine.complete_project(profile, index, payload.lore_summary)
    if not success: raise HTTPException(status_code=400, detail=msg)
    return {"status": "Project Completed", "message": msg}

# ==========================================
# 6. 🕸️ NETWORK MAP MODULE
# ==========================================

@app.get("/graph/{profile}")
def get_network_graph(profile: str):
    """Retrieves the React Flow node/edge graph from the backend."""
    return engine.generate_network_graph(profile)

@app.post("/graph/{profile}")
def save_graph_positions(profile: str, payload: GraphUpdate):
    """Saves the X/Y coordinates of nodes so the layout persists."""
    state = engine.get_world_state(profile)
    
    # 1. Map Cast by ID (The new source of truth)
    cast_map = {c["id"]: c for c in state.get("Cast", [])}
    
    # 2. Get Assets List (for index-based mapping)
    assets = state.get("Assets", [])
    
    for update in payload.updates:
        uid = update["id"]
        if "position" not in update: continue
        
        new_pos = {"x": update["position"]["x"], "y": update["position"]["y"]}
        
        # Update Character
        if uid in cast_map:
            cast_map[uid]["ui_pos"] = new_pos
            
        # Update Asset (Map "asset_0" -> index 0)
        elif uid.startswith("asset_"):
            try:
                idx = int(uid.split("_")[1])
                if 0 <= idx < len(assets):
                    assets[idx]["ui_pos"] = new_pos
            except: pass
            
    engine.save_world_state(profile, state)
    return {"status": "Positions Saved"}


# ==========================================
# 7. 🗣️ REACTION TOOL MODULE
# ==========================================

@app.post("/reaction/generate/{profile}")
def generate_faction_reaction(profile: str, payload: ReactionRequest):
    """Generates a faction response to a scene."""
    try:
        success, result = engine.generate_reaction_for_scene(
            profile, 
            payload.scene_file, 
            payload.faction, 
            public_only=payload.public_only, 
            format_style=payload.format_style,
            custom_instructions=payload.custom_instructions,
            timeline=payload.timeline
        )
        if not success:
            raise HTTPException(status_code=400, detail=result)
        return {"status": "Success", "content": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reaction/history/{profile}")
def get_all_reactions(profile: str):
    """Fetches all past faction reactions from the database."""
    try:
        return engine.get_all_faction_memories(profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/reaction/edit/{profile}/{reaction_id}")
def edit_reaction(profile: str, reaction_id: int, payload: EditReactionRequest):
    """Updates the text of a specific reaction in the database."""
    try:
        engine.update_faction_reaction(profile, reaction_id, payload.new_text, payload.new_faction)
        return {"status": "Updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reaction/delete/{profile}/{reaction_id}")
def delete_reaction(profile: str, reaction_id: int):
    """Hard deletes a reaction from the database."""
    try:
        engine.delete_faction_reaction(profile, reaction_id)
        return {"status": "Deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reaction/undo/{profile}")
def undo_last_reaction(profile: str, payload: UndoReactionRequest):
    """Rewinds the last reaction: Removes text and wipes memory."""
    try:
        # 1. Text Rollback
        file_success, file_msg = engine.undo_last_reaction_text(profile, payload.scene_file, payload.faction)
        
        # 2. Database Wipe (Delegate to Engine)
        engine.delete_last_faction_reaction(profile, payload.faction)
        
        return {
            "status": "Success" if file_success else "Partial",
            "file_message": file_msg,
            "db_message": "Memory Wiped"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 8. 📚 COMPILER MODULE
# ==========================================

@app.post("/compiler/compile/{profile}")
def compile_text_preview(profile: str, payload: CompileRequest):
    """Compiles selected scenes into a single text document."""
    try:
        raw_text = engine.compile_manuscript(profile, payload.filenames)
        return {"text": raw_text, "status": "Compiled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compiler/export/{profile}/{file_format}")
def export_binary_manuscript(profile: str, file_format: str, payload: CompileRequest):
    """Streams binary file (PDF/EPUB) to the client."""
    try:
        formats = engine.compile_formatted_manuscript(profile, payload.filenames)
        
        data = None
        media_type = ""
        filename = ""

        # 1. Generate Timestamp (YYYYMMDD_HHMMSS)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if file_format.lower() == "pdf":
            data = formats.get("pdf")
            media_type = "application/pdf"
            # 2. Append Timestamp to Filename
            filename = f"{profile}_Manuscript_{timestamp}.pdf"
        elif file_format.lower() == "epub":
            data = formats.get("epub")
            media_type = "application/epub+zip"
            # 2. Append Timestamp to Filename
            filename = f"{profile}_Manuscript_{timestamp}.epub"
        
        if not data:
             raise HTTPException(status_code=500, detail=f"Engine failed to generate {file_format} data.")

        return Response(
            content=data, 
            media_type=media_type, 
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"Export Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 9. ⚙️ SETTINGS MODULE
# ==========================================

@app.get("/settings/models")
def get_available_ai_models():
    """Dynamically lists available models (cached)."""
    try:
        return engine.list_available_models_all()
    except Exception as e:
        print(f"Model List Error: {e}")
        return ["gemini-1.5-flash", "gpt-4o"] # Fallback

@app.get("/settings/{profile}")
def get_system_settings(profile: str):
    """Retrieves global configuration."""
    return engine.get_story_settings(profile)

@app.post("/settings/update/{profile}")
def update_system_setting(profile: str, key: str, value: str):
    """Updates a single configuration key."""
    engine.update_story_setting(profile, key, value)
    return {"status": "Setting Updated"}