"""
Chronos Story Director - User Interface
=======================================
The Streamlit-based frontend application for the Chronos engine.
This module handles user interaction, visualizes world state via interactive
graphs, and orchestrates the narrative generation workflow.

Copyright (c) 2025 SirTonyEdgar
Licensed under the MIT License.
"""

import sqlite3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
import os
import io
import json
import glob
import datetime
import math
import uuid
import pandas as pd
import streamlit as st
import base64
from pypdf import PdfReader
from streamlit_agraph import agraph, Node, Edge, Config
import shutil

# Internal Engine Import
import backend as engine

# --- GLOBAL CONFIGURATION & ASSETS ---
st.set_page_config(
    page_title="Chronos Director", 
    page_icon="🕰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER: ASSET LOADER ---
def get_icon_path(filename):
    """Returns the local file path for Streamlit UI (st.image)."""
    return os.path.join("assets", filename)

def get_icon_base64(filename):
    """
    Converts a local image to a Base64 Data URI.
    REQUIRED for streamlit-agraph (Network Map) to display local images.
    """
    filepath = os.path.join("assets", filename)

    if not os.path.exists(filepath):
        return "https://img.icons8.com/?size=100&id=12116&format=png&color=000000"
        
    with open(filepath, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    return f"data:image/png;base64,{encoded}"

# Centralized Icon Repository to ensure consistency across Status and Network tabs
AVATAR_MANIFEST = {
    "Male": "male.png",
    "Female": "female.png",
    "Neutral": "neutral.png",
    "Villain": "villain.png",

    "Leader/Noble": "crown.png",
    "Official/Diplomat": "diplomat.png",
    "Wizard": "wizard.png",
    "Tech/Cyborg": "cyborg.png",
    "Soldier": "soldier.png",
    "Knight": "knight.png",
    "Merchant": "merchant.png",
    "Worker/Civilian": "worker.png",

    "Child": "child.png",
    "Student": "student.png",
    "Cat": "cat.png"
}

RELATIONSHIP_ICONS = {
    "Ally": "handshake.png",
    "Family": "family.png",
    "Enemy": "skull.png",
    "Rival": "fist-bump.png",
    "Love": "heart.png",
    "Mentor": "mentoring.png",
    "Subordinate/Vassal": "human.png",
    "Target": "target.png",
    "Resource/Item": "treasure-chest.png",
    "Vehicle/Transport": "wheel.png",      
    "Infrastructure/Base": "infrastructure.png",
    "Organization/Corp": "briefcase.png",
    "Weapon": "sword.png",
    "Technology": "technology.png",
    "Magic": "potion.png",
    "Wealth/Economy": "wealth.png",
    "Investment": "investment.png",
    "Intel/Secrets": "ancient-scroll.png",
    "Security/Defense": "shield.png",
    "Unknown": "question-mark.png"
}

# --- CACHING & UTILITIES ---
@st.cache_data(ttl=300) 
def get_cached_models():
    """Retrieves available LLM models with a 5-minute cache to reduce API latency."""
    return engine.list_available_models_all()

# --- SESSION STATE INITIALIZATION ---
if "active_profile" not in st.session_state:
    # Support deep linking via URL parameters (e.g. ?profile=MyStory)
    query_params = st.query_params
    url_profile = query_params.get("profile", None)
    if url_profile and url_profile in engine.list_profiles(): 
        st.session_state["active_profile"] = url_profile
    else: 
        st.session_state["active_profile"] = None

# --- LANDING PAGE (PROFILE SELECTION) ---
if st.session_state["active_profile"] is None:
    st.title("🕰️ Chronos Story Director")
    st.subheader("Select Profile")
    
    col_sel, col_new = st.columns(2)
    
    with col_sel:
        profs = engine.list_profiles()
        if profs:
            sel = st.selectbox("Existing Profiles", profs, label_visibility="collapsed")
            if st.button("Load Profile", type="primary"):
                st.session_state["active_profile"] = sel
                st.query_params["profile"] = sel
                st.rerun()
        else:
            st.info("No profiles found. Create one to begin.")

    with col_new:
        new_n = st.text_input("New Profile Name", placeholder="e.g. Project_Titan")
        if st.button("Create New"):
            if new_n:
                safe_name = new_n.strip().replace(" ", "_")
                engine.ensure_profile_structure(safe_name)
                st.session_state["active_profile"] = safe_name
                st.query_params["profile"] = safe_name
                st.rerun()
            else:
                st.error("Profile name cannot be empty.")
    
    st.stop() 

# --- MAIN APPLICATION LAYOUT ---
profile = st.session_state["active_profile"]
    
# --- NAVIGATION WITH PERSISTENCE ---
with st.sidebar:
    st.title("🕰️ Chronos")
    st.caption(f"Profile: **{profile}**")
    
    if st.button("🔄 Switch Profile"):
        st.session_state.clear() 
        st.query_params.clear()
        st.rerun()
    
    st.divider()

    url_page = st.query_params.get("page", "🎬 Scene Creator")

    nav_options = [
        "🎬 Scene Creator", "⚔️ War Room", "🕸️ Network Map", 
        "📊 World State Tracker", "🗄️ Knowledge Base", 
        "🗣️ Reaction Tool", "🧠 Co-Author Chat", "📚 Compiler", "⚙️ Settings"
    ]
    if "active_page" not in st.session_state:
        st.session_state.active_page = st.query_params.get("page", "🎬 Scene Creator")

    def sync_page_to_url():
        st.query_params["page"] = st.session_state.active_page

    page = st.radio(
        "Navigation", 
        nav_options, 
        key="active_page", 
        on_change=sync_page_to_url, 
        label_visibility="collapsed"
    )
    
    st.divider()

    # --- EXPORT ---
    with st.expander("📦 Export / Backup"):
        if st.button("Download Profile as ZIP"):
            shutil.make_archive(f"{profile}_backup", 'zip', engine.get_paths(profile)['root'])
            
            with open(f"{profile}_backup.zip", "rb") as f:
                st.download_button(
                    label="Click to Download",
                    data=f,
                    file_name=f"{profile}_Backup_{datetime.date.today()}.zip",
                    mime="application/zip"
                )

    st.caption("v14.2 - Adaptive Realism & Local Assets")

# ==========================================
# MODULE: SCENE CREATOR
# ==========================================
if page == "🎬 Scene Creator":
    st.header("🎬 Scene Creator")
    
    # Sync Sub-Tab with URL
    sc_tab_list = ["✍️ Write", "📖 Read", "✏️ Edit", "🗑️ Manage"]
    url_sc_tab = st.query_params.get("sc_tab", "✍️ Write")
    
    # Render Tabs
    tab_write, tab_read, tab_edit, tab_manage = st.tabs(sc_tab_list)
    
    with tab_write:
        # Load constraints and current state
        settings = engine.get_story_settings(profile)
        use_time = settings.get('use_time_system', 'true').lower() == 'true'
        current_state = engine.get_world_state(profile)

        # --- Auto-Detect Next Chapter ---
        next_ch = engine.get_next_chapter_number(profile)

        # Layout: Chapter | Title
        next_ch_hint = str(engine.get_next_chapter_number(profile))

        c_meta1, c_meta2 = st.columns([1, 4])
        with c_meta1:
            chapter_num = st.number_input("Chapter", min_value=1, value=None, placeholder=next_ch_hint, step=1, help="Used for file sorting (e.g. Ch01).")
        with c_meta2:
            title = st.text_input("Scene Title", placeholder="Optional (Auto-Generate if Empty)", key="sc_title")

        # Time System Controls
        if use_time:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                def_year_hint = str(engine.get_world_state(profile).get("Current_Year", None))
                year = st.number_input("Year", value=None, placeholder=def_year_hint, step=1, format="%d", key="sc_year")
            with c2: 
                date_str = st.text_input("Date", placeholder="e.g. March 6", key="sc_date")
            with c3: 
                time_str = st.text_input("Time", placeholder="Auto-Detect If Left Empty", key="sc_time")
        else:
            year, date_str, time_str = 0, "", ""
            st.caption("⏳ Time System is Disabled.")

        # Briefing & Context
        all_files = ["Auto (Last 3 Scenes)"] + engine.get_all_files_list(profile)
        context_files = st.multiselect("Transition From:", all_files, default=[], key="sc_context")
        brief = st.text_area("Scene Brief", height=300, key="sc_brief", placeholder="Describe the action...")
        
        # Privacy Control (Fog of War)
        auto_privacy = st.checkbox("🕵️ Enable Fog of War (Auto-Tag Private Scenes)", value=False, help="If enabled, the AI will wrap private conversations in [[PRIVATE]] tags.")

        if st.button("Generate Scene", type="primary", use_container_width=True, key="btn_gen_scene"):
            if not brief: 
                st.error("Scene Brief is required.")
            else:
                with st.spinner(f"Drafting Chapter {chapter_num}..."):
                    text, path = engine.generate_scene(
                        profile, chapter_num, year, date_str, time_str, title, brief,
                        context_files, use_fog_of_war=auto_privacy
                    )
                    st.success(f"Draft Saved: {os.path.basename(path)}")
                    st.rerun()

    with tab_read:
        all_files = engine.get_all_files_list(profile)
        if all_files:
            sel = st.selectbox("Select File:", all_files, key="sc_read_select")
            if sel: 
                st.markdown(engine.read_file_content(profile, sel).replace("\n", "  \n"))

    with tab_edit:
        paths = engine.get_paths(profile)
        files = [os.path.basename(f) for f in glob.glob(os.path.join(paths['output'], "*.txt"))]
        files.sort(reverse=True)

        if files:
            sel = st.selectbox("Edit File:", files, key="sc_edit_select")
            cont = engine.read_file_content(profile, sel)
            
            new_c = st.text_area("Editor", value=cont, height=600, key=f"editor_{sel}")

            if st.button("Save Changes", key=f"save_{sel}"):
                engine.save_edited_scene(profile, sel, new_c)
                st.success(f"Saved: {sel}")

    with tab_manage:
        paths = engine.get_paths(profile)
        files = [os.path.basename(f) for f in glob.glob(os.path.join(paths['output'], "*.txt"))]
        if files:
            sel = st.selectbox("Delete File:", files, key="sc_del_select")
            if st.button("Permanently Delete", type="secondary", key="btn_del_file"):
                engine.delete_specific_scene(profile, sel)
                st.rerun()

# ==========================================
# MODULE: WAR ROOM
# ==========================================
elif page == "⚔️ War Room":
    st.header("⚔️ War Room")
    st.caption("Monte Carlo Strategy Simulator")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📍 Operational Parameters")
        scenario = st.text_area("Proposed Action / Strategy", height=200, placeholder="Example: Hostile takeover...")
        
        if st.button("🔴 Run Simulation", type="primary"):
            if not scenario:
                st.error("Define a scenario first.")
            else:
                with st.spinner("Running Strategic Simulations..."):
                    report = engine.run_war_room_simulation(profile, scenario)
                    st.session_state['last_sim_report'] = report
                    st.session_state['last_sim_scenario'] = scenario
    
    with c2:
        st.info("ℹ️ **System Logic:**\nThe AI cross-references your **Allies**, **Assets**, and **World Rules**.")

    if 'last_sim_report' in st.session_state:
        st.divider()
        st.subheader("📊 Simulation Report")
        st.markdown(st.session_state['last_sim_report'])
        
        with st.popover("💾 Save to Plans"):
            plan_name = st.text_input("Plan Name", value="Operation: " + st.session_state['last_sim_scenario'][:20])
            if st.button("Confirm Save"):
                full_content = f"SCENARIO:\n{st.session_state['last_sim_scenario']}\n\nREPORT:\n{st.session_state['last_sim_report']}"
                engine.add_fragment(profile, plan_name, full_content, "Plan")
                st.success("Saved to Knowledge Base!")

# ==========================================
# MODULE: NETWORK MAP
# ==========================================
elif page == "🕸️ Network Map":
    st.header("🕸️ Network Map")
    
    # Load State
    if 'dashboard_state' not in st.session_state:
        st.session_state['dashboard_state'] = engine.get_world_state(profile)
    current_state = st.session_state['dashboard_state']

    # Controls & Interaction Lock
    c1, c2 = st.columns([3, 1])
    with c1:
        st.info("💡 To add new characters or change connections, go to the **Status & Assets** page.")
    with c2:
        enable_interaction = st.checkbox(
            "🔓 Unlock Zoom/Pan", 
            value=False, 
            help="Check this to enable dragging and zooming. Keep unchecked to scroll the page easily."
        )

    nodes = []
    edges = []
    font_style = {'color': 'white', 'strokeWidth': 2, 'strokeColor': 'black', 'size': 24}
    
    # --- ID COLLISION TRACKER ---
    existing_ids = set()

    edge_font = {'size': 22, 'color': 'black', 'align': 'middle', 'strokeWidth': 1.5, 'strokeColor': 'white'}

    # Build Nodes
    p_data = current_state.get("Protagonist Status", {})
    p_name = p_data.get("Name", "Protagonist")
    p_icon_key = p_data.get("Icon", "Male")

    p_icon_filename = AVATAR_MANIFEST.get(p_icon_key, "male.png")
    p_icon_url = get_icon_base64(p_icon_filename)

    nodes.append(Node(
        id="MAIN", label=p_name, size=50, shape="image", 
        image=p_icon_url,
        font=font_style, x=0, y=0, fixed=True
    ))

    existing_ids.add("MAIN")

    allies = current_state.get("Allies", [])
    assets = current_state.get("Assets", [])
    total = len(allies) + len(assets)
    radius = 400 
    
    def get_coords(idx, total, r):
        if total == 0: return 0, 0
        angle = (2 * math.pi * idx) / total
        return int(r * math.cos(angle)), int(r * math.sin(angle))

    current_idx = 0
    
    # Combine Global Icon Maps
    FULL_ICON_MAP = {**RELATIONSHIP_ICONS, **AVATAR_MANIFEST}

    # --- ALLY LOOP ---
    for ally in allies:
        # Check 'Name' key, fallback to 'Unknown'
        raw_name = ally.get("Name", "Unknown Ally")
        
        # Deduplicate ID (e.g. if two allies are named "Guard", make "Guard_1")
        node_id = raw_name
        counter = 1
        while node_id in existing_ids:
            node_id = f"{raw_name}_{counter}"
            counter += 1
        existing_ids.add(node_id)

        filename = FULL_ICON_MAP.get(ally.get("Icon", "Ally"), "handshake.png")
        icon_url = get_icon_base64(filename)
        
        lx, ly = get_coords(current_idx, total, radius)
        current_idx += 1
        
        nodes.append(Node(id=node_id, label=raw_name, size=40, shape="image", image=icon_url, font=font_style, x=lx, y=ly, fixed=True))
        edges.append(Edge(source="MAIN", target=node_id, label=ally.get("Relation", "Ally"), color="#4CAF50", font=edge_font))

    # --- ASSET LOOP ---
    for asset in assets:
        raw_name = asset.get("Asset", asset.get("Name", "Unknown Item"))
        
        # Deduplicate ID
        node_id = raw_name
        counter = 1
        while node_id in existing_ids:
            node_id = f"{raw_name}_{counter}"
            counter += 1
        existing_ids.add(node_id)

        icon_key = asset.get("Icon", "Resource")
        filename = FULL_ICON_MAP.get(icon_key, "chest.png")
        icon_url = get_icon_base64(filename)

        lx, ly = get_coords(current_idx, total, radius)
        current_idx += 1
        
        nodes.append(Node(
            id=node_id, label=raw_name, size=35, shape="image", 
            image=icon_url,
            font=font_style, x=lx, y=ly, fixed=True
        ))
        edges.append(Edge(source="MAIN", target=node_id, label=asset.get("Type", "Resource"), color="#FFC107", font=edge_font))

    # Render
    if len(nodes) > 1:
        with st.container(border=True):
            config = Config(
                width="stretch", 
                height=750, 
                directed=True, 
                nodeHighlightBehavior=True, 
                highlightColor="#F7A7A6",
                collapsible=False,
                fit=True, 
                physics={"enabled": False},
                interaction={
                    "dragNodes": enable_interaction, 
                    "dragView": enable_interaction, 
                    "zoomView": enable_interaction
                }
            )
            agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.info("Graph is empty. Go to 'Status & Assets' to add Allies.")

# ==========================================
# MODULE: WORLD STATE TRACKER
# ==========================================
elif page == "📊 World State Tracker":
    st.header("📊 World State Tracker")
    
    if 'dashboard_state' not in st.session_state:
        st.session_state['dashboard_state'] = engine.get_world_state(profile)
    
    # Process batch analysis callbacks
    if 'temp_state' in st.session_state:
        st.session_state['dashboard_state'] = st.session_state.pop('temp_state')
        st.toast("State updated from AI Analysis!")

    current_state = st.session_state['dashboard_state']
    
    # --- 🛠️ AUTO-MIGRATION LOGIC ---
    if not current_state.get("Allies") and "Relations" in current_state:
        if "Allies" in current_state["Relations"]:
            current_state["Allies"] = current_state["Relations"]["Allies"]
            st.toast("Migrated Allies from nested 'Relations' object.", icon="📦")

    # --- AI ANALYSIS TOOLS ---
    with st.expander("🤖 AI Batch Analysis Tools", expanded=False):
        st.caption("Select Scenes, Lore, or Plans to auto-extract data (Dates, Allies, Assets).")
        
        all_readable_files = engine.get_all_files_list(profile)
        
        if all_readable_files:
            target_scenes = st.multiselect("Select Content to Analyze:", all_readable_files, default=[])
            
            if st.button("Analyze & Update State", width="stretch"):
                if target_scenes:
                    with st.spinner("Processing Context & Extracting Data..."):
                        combined = ""
                        for fname in target_scenes:
                            combined += f"\n=== {fname} ===\n{engine.read_file_content(profile, fname)}\n"
                        
                        st.session_state['temp_state'] = engine.analyze_state_changes(profile, combined)
                        st.rerun()

    st.divider()
    st.subheader("🏰 Tracker Dashboard")

    ws_tab_options = ["👤 Protagonist", "🚧 Projects", "🤝 Relations", "💰 Assets", "⚡ Skills", "🌐 Variables", "📝 JSON"]

    if "ws_active_tab" not in st.session_state:
        st.session_state.ws_active_tab = st.query_params.get("ws_tab", ws_tab_options[0])

    def sync_ws_tab():
        st.query_params["ws_tab"] = st.session_state.ws_active_tab
    
    selected_tab = st.segmented_control(
        "Dashboard Selection", 
        options=ws_tab_options, 
        key="ws_active_tab", 
        on_change=sync_ws_tab,
        label_visibility="collapsed"
    )
    
    # --- TAB: PROTAGONIST IDENTITY ---
    if selected_tab == "👤 Protagonist":
        p_data = current_state.get("Protagonist Status", {})
        if not isinstance(p_data, dict): p_data = {}
        
        # Sync with DB Settings
        sys_settings = engine.get_story_settings(profile)
        
        st.info("🎭 **Identity Protocol:** 'True Identity' accesses deep memory/lore. 'Current Name' is used for prose generation (e.g. handling disguises or titles).")
        
        c1, c2 = st.columns(2)
        
        # --- LEFT COLUMN: VISUALS & PERSONA ---
        with c1:
            st.subheader("📖 Narrative Persona")
            p_data["Name"] = st.text_input("Current Narrative Name", value=p_data.get("Name", "Unknown"), help="The specific name used in prose generation (e.g. 'The Red Ranger' vs 'Jason'). Change this to handle disguises.")
            p_data["Aliases"] = st.text_area("Known Aliases / Nicknames", value=p_data.get("Aliases", ""), help="Other names the AI should recognize as referring to the protagonist (e.g. 'Subject Zero, The Asset').", height=68)
            
            # Current Goal
            p_data["Goal"] = st.text_area("Current Goal", value=p_data.get("Goal", ""), height=100, help="The immediate driving motivation. The AI uses this to determine character actions in the War Room and Scenes.")

            # Protagonist Icon Selector
            current_icon = p_data.get("Icon", "Male")
            if current_icon not in AVATAR_MANIFEST: current_icon = "Male"
            p_data["Icon"] = st.selectbox("Map Avatar", options=list(AVATAR_MANIFEST.keys()), index=list(AVATAR_MANIFEST.keys()).index(current_icon), key="p_avatar")
            
            # Avatar Preview
            st.caption("Avatar Preview:")
            icon_filename = AVATAR_MANIFEST[p_data["Icon"]]
            st.image(get_icon_path(icon_filename), width=60)

        # --- RIGHT COLUMN: DATA & TIME ---
        with c2:
            st.subheader("🧠 Deep Memory")
            true_name = st.text_input("True Identity (Context)", value=sys_settings.get('protagonist', ''), help="The character's actual identity. This ensures the AI accesses the correct memories even if the narrative name changes.")
            
            # --- Smart Age System ---
            st.caption("⏳ Temporal Tracking")
            
            # Choose Mode
            default_mode = 0
            if "Birth_Year" in p_data: default_mode = 1
            
            age_mode = st.selectbox(
                "Tracking Mode", 
                ["Static Status (Manual)", "Birth Year (Auto-Calc)"], 
                index=default_mode,
                help="Use 'Static' for abstract states (e.g. 'Ancient'). Use 'Birth Year' to let the AI auto-calculate age based on the Scene Year."
            )
            
            if age_mode == "Birth Year (Auto-Calc)":
                # World Clock (Global State)
                curr_year = current_state.get("Current_Year", None) 
                c_y_input = st.number_input("Current World Year", value=curr_year, step=1, help="The current year in the narrative timeline.")
                
                # Birth Year
                b_year_val = p_data.get("Birth_Year", None)
                birth_year = st.number_input("Character Birth Year", value=b_year_val, step=1, help="Used to calculate age relative to the World Year.")

                p_data["Date of Birth"] = st.text_input("Full Date of Birth", value=p_data.get("Date of Birth", ""), placeholder="e.g. 1984-03-06")
                
                # The Math (Safety Check)
                if c_y_input is not None and birth_year is not None:
                    calc_age = c_y_input - birth_year
                    st.write(f"**Current Age:** {calc_age}")
                    p_data["Age"] = str(calc_age)
                    p_data["Birth_Year"] = birth_year
                    current_state["Current_Year"] = c_y_input
                else:
                    st.info("Pending Data: Enter years manually or run AI Analysis to calculate age.")
                
            else:
                # Manual Mode
                p_data["Age"] = st.text_input("Age / Status", value=str(p_data.get("Age", "Unknown")), help="Manual override string (e.g. '25', 'Immortal', 'Cryo-Frozen').")
                if "Birth_Year" in p_data: del p_data["Birth_Year"]

        current_state["Protagonist Status"] = p_data
        
        if st.button("💾 Update Identity", type="primary"):
            engine.save_world_state(profile, current_state)
            engine.update_story_setting(profile, 'protagonist', true_name)
            st.toast("Identity Updated.", icon="✅")
            st.rerun()

    # --- TAB: PROJECTS ---
    elif selected_tab == "🚧 Projects":
        st.info("🚧 **Long-Term Goals:** Projects track sustained efforts over time (e.g., 'Building a Base', 'Researching a Cure'). The AI checks the 'Specifications' and 'Progress' to determine if you have the capability to perform advanced actions.")
        projects = current_state.get("Projects", [])
        
        # New Project Creator
        with st.expander("🚀 Launch New Project", expanded=False):
            with st.form("new_proj_form"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    n_name = st.text_input("Project Name", placeholder="e.g. Kernel Logic", help="Unique identifier for this long-term effort.")
                with c2:
                    n_desc = st.text_input("Objective", placeholder="e.g. Create predictive RAM algorithm", help="The success condition. What does '100% Progress' look like?")
                n_specs = st.text_area("Initial Specifications", height=100, help="Technical constraints or resources required. The AI checks this during simulations.")
                
                if st.form_submit_button("Initialize Project"):
                    engine.add_project(profile, n_name, n_desc, n_specs)
                    # Force reload to clear cache
                    if 'dashboard_state' in st.session_state:
                        del st.session_state['dashboard_state']
                    st.rerun()

        st.divider()

        # Active Projects Tracker
        if not projects: 
            st.info("No active projects. Start one above.")
        else:
            for i, proj in enumerate(projects):
                # Safe Data Retrieval
                p_name = proj.get('Name', 'Unnamed Project')
                p_desc = proj.get('Description', 'No objective defined.')
                p_prog = proj.get('Progress', 0)
                p_specs = proj.get('Features_Specs', '')

                # --- READ-ONLY VIEW ---
                st.markdown(f"### 🏗️ {p_name}")
                st.caption(f"**Objective:** {p_desc}")
                
                st.write(f"**Status:** {p_prog}% Complete")
                
                st.progress(p_prog / 100)
                
                # --- EDITING INTERFACE ---
                with st.expander(f"⚙️ Edit / Manage: {p_name}"):
                    # Editable Name & Objective
                    new_name = st.text_input("Project Name", value=p_name, key=f"name_{i}")
                    new_desc = st.text_area("Objective", value=p_desc, key=f"desc_{i}", height=150)
                    
                    # Editable Specs
                    new_specs = st.text_area("Technical Specs", value=p_specs, key=f"spec_{i}", height=300)
                    
                    # Editable Progress
                    new_prog = st.slider("Progress %", 0, 100, p_prog, key=f"prog_{i}")
                    
                    # Action Buttons
                    c_up, c_arc, c_del = st.columns([1, 1, 1])
                    
                    # Save / Update
                    if c_up.button("💾 Save Changes", key=f"upd_{i}", use_container_width=True):
                        engine.update_project(profile, i, new_prog, new_specs, new_name=new_name, new_desc=new_desc)
                        if 'dashboard_state' in st.session_state:
                            del st.session_state['dashboard_state']
                        st.success("Saved!")
                        st.rerun()
                    
                    # Complete / Archive Popover
                    with c_arc.popover("✅ Complete", use_container_width=True):
                        st.subheader("Archive Project")
                        st.caption("This moves project details into the Knowledge Base and removes it from the active tracker.")
                        
                        # --- Archive Destination Choice ---
                        archive_target = st.radio(
                            "Archive to:", 
                            ["Lore", "Fact"], 
                            index=1, 
                            horizontal=True, 
                            help="Lore is for world building. Fact is for a specific historical event.",
                            key=f"archive_radio_{i}"
                        )
                        
                        final_hist = st.text_area("Final Summary", value=f"Final Result: {new_name}. Specs: {new_specs}", height=150, key=f"hist_{i}")
                        
                        if st.button("Confirm & Archive", key=f"fin_{i}", type="primary", use_container_width=True):
                            engine.complete_project(profile, i, final_hist, target_category=archive_target)
                            st.balloons()
                            st.rerun()

                    # Delete
                    with c_del.popover("🗑️ Delete", use_container_width=True):
                        st.error("This cannot be undone.")
                        if st.button("Confirm Deletion", key=f"del_{i}", type="primary"):
                            del current_state["Projects"][i]
                            engine.save_world_state(profile, current_state)
                            st.rerun()
                
                if i < len(projects) - 1:
                    st.divider()

    # --- TAB: RELATIONS ---
    elif selected_tab == "🤝 Relations":
        st.info("🤝 **Social Web:** Defines who helps or hinders you. The AI uses 'Loyalty' to calculate betrayal risks and 'Notes' to remember leverage/secrets during social interactions.")
        target_list = current_state.get("Allies", [])
        df_allies = pd.DataFrame(target_list)
        
        if df_allies.empty:
            df_allies = pd.DataFrame(columns=["Name", "Relation", "Loyalty", "Notes", "Icon"])
        if "Icon" not in df_allies.columns:
            df_allies["Icon"] = "Ally"

        # Legend
        with st.expander("🖼️ Icon Reference"):
            st.caption("These icons determine the visual representation on the **Network Map**.")
            cols = st.columns(9)
            i = 0
            for name, url in RELATIONSHIP_ICONS.items():
                with cols[i % 9]:
                    st.image(get_icon_path(url), width=40)
                    st.caption(name)
                i += 1

        current_state["Allies"] = st.data_editor(
            df_allies,
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("Name", required=True, width="medium"),
                "Relation": st.column_config.TextColumn("Relation", default="Ally", help="Role relative to Protagonist (e.g. 'Rival', 'Mentor')."),
                "Loyalty": st.column_config.NumberColumn("Loyalty %", min_value=0, max_value=100, default=50, help="0 = Betrayal Imminent, 100 = Unshakeable."),
                "Icon": st.column_config.SelectboxColumn("Map Icon", options=list(RELATIONSHIP_ICONS.keys()), required=True, default="Ally"),
                "Notes": st.column_config.TextColumn("Notes", width="large", help="Private notes on leverage, secrets, or status.")
            },
            key="editor_allies"
        ).to_dict(orient="records")

    # --- TAB: ASSETS ---
    elif selected_tab == "💰 Assets":
        st.info("💰 **Inventory & Resources:** Physical or abstract tools. These provide 'Narrative Permission' (e.g. you cannot fly to Paris without a 'Private Jet' asset) and boost your power in the War Room.")
        assets_list = current_state.get("Assets", [])
        df_assets = pd.DataFrame(assets_list)
        if df_assets.empty:
            df_assets = pd.DataFrame(columns=["Asset", "Type", "Status", "Value", "Notes", "Icon"])

        if "Icon" not in df_assets.columns:
            df_assets["Icon"] = "Resource"
        if "Notes" not in df_assets.columns: df_assets["Notes"] = ""
        
        with st.expander("🖼️ Icon Reference"):
            st.caption("Select an icon to represent this asset on the Network Map.")
            cols = st.columns(9)
            i = 0
            for name, url in RELATIONSHIP_ICONS.items():
                with cols[i % 9]:
                    st.image(get_icon_path(url), width=40)
                    st.caption(name)
                i += 1

        current_state["Assets"] = st.data_editor(
            df_assets,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Asset": st.column_config.TextColumn("Asset Name", required=True),
                "Type": st.column_config.TextColumn("Type", default="Financial", help="Category (e.g. 'Weapon', 'Property', 'Information')."),
                "Status": st.column_config.TextColumn("Status", default="Active", help="Current condition (e.g. 'Damaged', 'Hidden', 'Liquidated')."),
                "Value": st.column_config.TextColumn("Value/Power", default="Unknown", help="Narrative weight or monetary worth."),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
                "Icon": st.column_config.SelectboxColumn("Map Icon", options=list(RELATIONSHIP_ICONS.keys()), required=True, default="Resource")
            },
            key="editor_assets"
        ).to_dict(orient="records")

    # --- TAB: SKILLS ---
    elif selected_tab == "⚡ Skills":
        st.info("⚡ **Competence Modifiers:** Capabilities that unlock specific solutions. If you have 'Hacking', the AI allows cyber-warfare options; if you lack it, those options fail.")
        skills = current_state.get("Skills", [])

        normalized_skills = []
        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, str):
                    normalized_skills.append({"Skill": s, "Description": ""})
                elif isinstance(s, dict):
                    normalized_skills.append(s)
        
        current_state["Skills"] = st.data_editor(
            normalized_skills,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Skill": st.column_config.TextColumn("Skill Name", required=True, width="medium"),
                "Description": st.column_config.TextColumn("Mechanics / Details", width="large", help="How this skill affects scene outcomes or roll modifiers.")
            },
            key="editor_skills"
        )

    # --- TAB: WORLD VARIABLES ---
    elif selected_tab == "🌐 Variables":
        st.info("🌐 **World Mechanics (Variables):** Abstract global states (e.g., 'Defcon Level', 'Corruption', 'Mana'). The AI reads these values to enforce the 'Logic/Consequence Rule' on narrative outcomes.")
        
        current_vars = [v for v in current_state.get("World Variables", []) if isinstance(v, dict)]
        df_vars = pd.DataFrame(current_vars) if current_vars else pd.DataFrame(columns=["Name", "Value", "Mechanic"])
        
        # Normalize Data (Handle legacy lists)
        normalized_vars = []
        if current_vars:
            for v in current_vars:
                if isinstance(v, dict): normalized_vars.append(v)
        
        # FORCE SCHEMA
        df_vars = pd.DataFrame(normalized_vars)
        if df_vars.empty:
            df_vars = pd.DataFrame(columns=["Name", "Value", "Mechanic"])

        # Render Editor
        current_state["World Variables"] = st.data_editor(
            df_vars,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("Variable Name", required=True, width="medium"),
                "Value": st.column_config.TextColumn("Current State", width="small", help="The current level/status (e.g. '75%', 'Critical')."),
                "Mechanic": st.column_config.TextColumn("Logic / Consequence Rule", width="large", help="The logical rule the AI follows (e.g. 'If Value > 50, police become hostile').")
            },
            key="editor_vars"
        ).to_dict(orient="records")

    # --- TAB: RAW JSON ---
    elif selected_tab == "📝 JSON":
        raw_text = st.text_area("Raw JSON", value=json.dumps(current_state, indent=4), height=400)
        if st.button("Override from JSON"):
            try:
                current_state = json.loads(raw_text)
                st.session_state['dashboard_state'] = current_state
                st.rerun()
            except Exception as e:
                st.error(f"JSON Error: {e}")

    st.divider()
    if st.button("💾 Save State to Disk", type="primary", width="stretch"):
        engine.save_world_state(profile, current_state)
        st.toast("World State Saved!", icon="✅")

# ==========================================
# MODULE: KNOWLEDGE BASE
# ==========================================
elif page == "🗄️ Knowledge Base":
    st.header("🗄️ Knowledge Base")

    kb_tab_options = ["📜 Lore", "📏 Rules", "🗺️ Plans", "📌 Facts", "🚫 Spoilers"]

    if "kb_active_tab" not in st.session_state:
        url_val = st.query_params.get("tab", "📜 Lore")
        st.session_state.kb_active_tab = url_val if url_val in kb_tab_options else "📜 Lore"

    def sync_kb_tab():
        st.query_params["tab"] = st.session_state.kb_active_tab

    selected_kb = st.segmented_control(
        "Knowledge Category", 
        options=kb_tab_options, 
        key="kb_active_tab", 
        on_change=sync_kb_tab,
        label_visibility="collapsed"
    )

    def render_smart_editor(doc_type, label, icon="📜", help_text=""):
        """Standardized editor with search, rename, height control, and custom hints."""
        if help_text:
            st.info(help_text)

        frags = engine.get_fragments(profile, doc_type)

        s_query = st.text_input(
            f"Search {label}", 
            placeholder=f"Filter {label} content...", 
            key=f"search_{doc_type}"
        )
        
        filtered = [f for f in frags if s_query.lower() in f[2].lower()] if s_query else frags

        with st.container(height=500):
            if not filtered:
                st.caption("No matching fragments found.")
            for fid, fname, cont, _ in filtered:
                with st.expander(f"{icon} {fname}"):
                    with st.form(f"edit_{doc_type}_{fid}"):
                        new_name = st.text_input("Label", value=fname)
                        new_cont = st.text_area("Content", value=cont, height=300)
                        
                        c1, c2 = st.columns(2)
                        if c1.form_submit_button("💾 Save Changes", use_container_width=True):
                            engine.update_fragment(profile, fid, new_cont)
                            engine.rename_fragment(profile, fid, new_name)
                            st.success("Updated!")
                            st.rerun()
                        if c2.form_submit_button("🗑️ Delete", use_container_width=True):
                            engine.delete_fragment(profile, fid)
                            st.rerun()
        
        st.divider()
        with st.form(f"add_new_{doc_type}_manual", clear_on_submit=True):
            st.caption(f"Add New {label}")
            n = st.text_input("Title")
            c = st.text_area("Content", height=150)
            if st.form_submit_button("Create", type="primary", use_container_width=True): 
                if c:
                    engine.add_fragment(profile, n if n else "Untitled", c, doc_type)
                    st.rerun()
                else:
                    st.error("Content cannot be empty.")

    if selected_kb == "📜 Lore":
        render_smart_editor(
            "Lore", "Lore", "📜", 
            help_text="📖 **Story Bible (Background):** Permanent history, geography, and character backstories. The AI consults this for context but knows these events happened in the *past*."
        )
    
    elif selected_kb == "📏 Rules":
        render_smart_editor(
            "Rulebook", "Rules", "📏", 
            help_text="📏 **World Physics (Immutable):** Hard laws of your universe (Magic Systems, FTL Physics, RPG Stats). The AI treats these as absolute constraints that *cannot* be broken."
        )
    
    elif selected_kb == "🗺️ Plans":
        render_smart_editor(
            "Plan", "Plans", "🗺️", 
            help_text="🗺️ **Future Roadmap (Context):** Plot points, villain schemes, or strategic goals that *haven't happened yet*. The AI uses this to foreshadow events without assuming they are current reality."
        )

    elif selected_kb == "📌 Facts":
        render_smart_editor(
            "Fact", "Facts", "📌", 
            help_text="📌 **Established Truths (Current):** Immediate narrative facts established in recent scenes (e.g., 'The King is dead', 'The base is destroyed'). These override older Lore."
        )
                
    elif selected_kb == "🚫 Spoilers":
        st.error("🚫 **Banned Content (The Anti-Prompt):** Concepts, twists, or names the AI is explicitly FORBIDDEN from mentioning until you decide it's time.")
        
        frags = engine.get_fragments(profile, "Spoiler")
        if not frags:
            st.info("No active spoilers.")
            
        for fid, fname, cont, _ in frags:
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                st.markdown(f"**STOP:** `{cont}`")
            with c2:
                if st.button("🗑️", key=f"ds_{fid}", help="Remove ban"): 
                    engine.delete_fragment(profile, fid)
                    st.rerun()
                    
        st.divider()
        with st.form("add_new_spoiler", clear_on_submit=True):
            ns = st.text_input("New Banned Term / Secret")
            if st.form_submit_button("Ban Term", type="primary", use_container_width=True): 
                if ns:
                    engine.add_fragment(profile, "Spoiler_Alert", ns, "Spoiler")
                    st.rerun()

# ==========================================
# MODULE: REACTION TOOL
# ==========================================
elif page == "🗣️ Reaction Tool":
    st.header("🗣️ Faction Reaction Engine")
    files = engine.get_all_files_list(profile)
    
    # Template Library for Sub-Options
    REACTION_TEMPLATES = {
        "👤 Individual / Personal": [
            "Internal Monologue / Private Thoughts",
            "Personal Diary / Journal Entry",
            "Direct Speech / Live Reaction",
            "Private Letter / Correspondence",
            "Prayer / Meditation / Communion"
        ],
        "🏛️ Political / Bureaucratic": [
            "Official Decree / Executive Order",
            "Senate / Council Debate",
            "Diplomatic Cable / Envoy Message",
            "Internal Memo / Briefing Document",
            "Propaganda Broadcast / Public Statement"
        ],
        "⚔️ Military / Tactical": [
            "Combat Report / Sitrep (Situation Report)",
            "Strategy Meeting / War Room",
            "Radio Chatter / Field Comms",
            "Officer's Log / Captain's Log",
            "Soldier's Gossip / Barracks Talk"
        ],
        "🕵️ Underground / Criminal": [
            "Thieves' Cant / Code Words",
            "Black Market Transaction Log",
            "Encrypted Channel / Dark Web Post",
            "Smuggler's Rumor / Tavern Whisper",
            "Anonymous Tip / Leak"
        ],
        "📢 Public Discourse / Media": [
            "News Front Page / Headline",
            "Social Media Feed / Viral Post",
            "Town Crier / Public Announcement",
            "Pundit Commentary / Opinion Piece",
            "Commoner's Gossip / Watercooler Talk"
        ],
        "🏢 Corporate / Economic": [
            "Shareholder Report / Quarterly Earnings",
            "Boardroom Meeting Minutes",
            "Sales Pitch / Advertisement",
            "Trade Guild Ledger / Manifest",
            "Worker's Union Meeting"
        ],
        "🔬 Scientific / Arcane": [
            "Lab Report / Research Log",
            "Medical Diagnosis / Autopsy",
            "Wizard's Grimoire / Spell Notes",
            "AI System Log / Debug Output",
            "Archaeological Discovery Note"
        ],
        "✨ Custom / Specific": ["Manual Input"]
    }

    if files:
        # Context Selection
        c1, c2 = st.columns([1, 1])
        with c1:
            target_scene = st.selectbox("Select Context Scene:", files)
        with c2:
            target_faction = st.text_input("Target Faction/Character:", placeholder="e.g. The peasantry, forum post, The Galactic Senate")

        # Format Selection
        c3, c4 = st.columns([1, 1])
        with c3:
            selected_category = st.selectbox("Perspective / Era", list(REACTION_TEMPLATES.keys()))
        
        with c4:
            available_formats = REACTION_TEMPLATES[selected_category]
            selected_format = st.selectbox("Format / Medium", available_formats)

        # Custom Format Override
        final_style_instruction = selected_format
        if selected_format == "Manual Input":
            final_style_instruction = st.text_input("Describe Custom Format:", placeholder="e.g. Telepathic dream sequence")

        # --- Additional Prompt / Specific Instructions ---
        extra_prompt = st.text_area(
            "Additional Instructions (Optional)", 
            placeholder="e.g. Make them sounds extremely skeptical, or focus on their fear of the protagonist.",
            help="Further guide the AI's tone or focus for this specific reaction."
        )

        # Fog of War Control
        st.caption("---")
        is_public = st.checkbox("👁️ Public Knowledge Only", value=True, help="If checked, the AI ignores secrets/internal thoughts and only reacts to what is visible.")
        
        st.divider()
        
        col_sim, col_undo = st.columns([4, 1])

        with col_sim:
            # THE SIMULATE BUTTON
            if st.button("Simulate Reaction", type="primary", use_container_width=True):
                if not target_faction:
                    st.error("Please specify a Faction.")
                else:
                    full_style_prompt = f"{selected_category} -> {final_style_instruction}"
                    
                    with st.spinner(f"Simulating ({final_style_instruction})..."):
                        success, res = engine.generate_reaction_for_scene(
                            profile, target_scene, target_faction, 
                            public_only=is_public, 
                            format_style=full_style_prompt,
                            custom_instructions=extra_prompt
                        )
                        
                        if success: 
                            st.success("Reaction Generated & Saved!")
                            st.markdown(f"### Output Preview")
                            st.write(res)
                        else: 
                            st.error(res)

        with col_undo:
            if st.button("Undo ⎌", help="Deletes last memory AND removes text from file.", use_container_width=True):
                if not target_faction:
                    st.error("Target Faction required.")
                else:
                    file_success, file_msg = engine.undo_last_reaction_text(profile, target_scene, target_faction)

                    try:
                        conn = sqlite3.connect(engine.get_paths(profile)['db'])
                        c = conn.cursor()
                        c.execute("DELETE FROM faction_memory WHERE id = (SELECT MAX(id) FROM faction_memory WHERE faction_name = ?)", (target_faction,))
                        conn.commit()
                        conn.close()
                        db_msg = "Memory Wiped"
                    except Exception as e:
                        db_msg = f"DB Error: {e}"

                    if file_success:
                        st.success(f"Rewind Complete: {file_msg} + {db_msg}")
                    else:
                        st.warning(f"Partial Rewind: {db_msg}. File warning: {file_msg} (You may need to delete text manually).")

# ==========================================
# MODULE: CO-AUTHOR CHAT
# ==========================================
elif page == "🧠 Co-Author Chat":
    st.header("🧠 Co-Author Interface")
    
    with st.sidebar:
        st.divider()
        if st.button("🗑️ Clear History"):
            engine.clear_chat_history(profile)
            st.rerun()
            
    history = engine.get_chat_history(profile)
    for msg in history: 
        st.chat_message(msg["role"]).markdown(msg["content"])
        
    if p := st.chat_input("Ask about lore, plot, or mechanics..."):
        st.chat_message("user").markdown(p)
        engine.save_chat_message(profile, "user", p)
        
        with st.spinner("Consulting Knowledge Base..."):
            res = engine.run_chat_query(profile, p)
        
        st.chat_message("assistant").markdown(res)
        engine.save_chat_message(profile, "assistant", res)
        st.rerun()
        
    if history and history[-1]["role"] == "assistant":
        st.divider()
        with st.expander("💾 Save Response as Draft"):
            def_name = f"Draft_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
            s_name = st.text_input("Filename:", value=def_name)
            if st.button("Save to Scenes"):
                engine.save_edited_scene(profile, s_name, history[-1]["content"])
                st.success("Draft Saved!")

# ==========================================
# MODULE: COMPILER
# ==========================================
elif page == "📚 Compiler":
    st.header("📚 Manuscript Compiler")
    st.caption("Compile scenes into distribution-ready formats (PDF/EPUB).")
    
    all_scenes = engine.get_all_files_list(profile)
    
    # Scene Selector
    # Scene Selector
    selected = st.multiselect(
        "Select Scenes to compile (Ordered):", 
        all_scenes, 
        default=[]
    )
    
    if selected:
        st.divider()
        st.subheader("📦 Export Options")
        
        c1, c2, c3 = st.columns(3)
        
        # Option 1: Raw Text
        with c1:
            if st.button("📝 Compile Plain Text"):
                text = engine.compile_manuscript(profile, selected)
                st.download_button("Download .txt", text, f"{profile}_Manuscript.txt")
                st.download_button("Download .md", text, f"{profile}_Manuscript.md")

        # Option 2: Formatted Binaries
        with c2:
            if st.button("📕 Generate Book (PDF/EPUB)"):
                with st.spinner("Typesetting document layout..."):
                    formats = engine.compile_formatted_manuscript(profile, selected)
                    
                    if formats["pdf"]:
                        st.download_button(
                            label="Download PDF (Print)",
                            data=formats["pdf"],
                            file_name=f"{profile}_Manuscript.pdf",
                            mime="application/pdf"
                        )
                    
                    if formats["epub"]:
                        st.download_button(
                            label="Download EPUB (E-Reader)",
                            data=formats["epub"],
                            file_name=f"{profile}_Manuscript.epub",
                            mime="application/epub+zip"
                        )
                    
                    if not formats["pdf"] and not formats["epub"]:
                        st.error("Formatting failed. Please check server logs for encoding errors.")

# ==========================================
# MODULE: SETTINGS
# ==========================================
elif page == "⚙️ Settings":
    st.header("⚙️ System Configuration")
    
    curr = engine.get_story_settings(profile)
    world_state = engine.get_world_state(profile)
    available_models = get_cached_models()
    
    with st.form("global_settings_form"):
        st.subheader("🌍 World Mechanics & AI Configuration")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**🧠 AI Model Routing**")

            safe_default = available_models[0] if available_models else "No Models Detected"
            def get_idx(val, opts): return opts.index(val) if val in opts else 0
            
            m_scene = st.selectbox(
                "Scene Writer", 
                available_models, 
                index=get_idx(curr.get('model_scene', safe_default), available_models),
                help="Generates the main story prose and narrative content."
            )
            
            m_chat = st.selectbox(
                "Co-Author", 
                available_models, 
                index=get_idx(curr.get('model_chat', safe_default), available_models),
                help="Handles the Chat interface for brainstorming, lore Q&A, and plot planning."
            )
            
            m_react = st.selectbox(
                "Reaction Engine", 
                available_models, 
                index=get_idx(curr.get('model_reaction', safe_default), available_models),
                help="Simulates faction reactions (Reaction Tool)."
            )
            
            m_analysis = st.selectbox(
                "Logic & Strategy Engine", 
                available_models, 
                index=get_idx(curr.get('model_analysis', safe_default), available_models), 
                help="Handles the deep reasoning for War Room Simulations and World State Analysis."
            )

            m_retrieval = st.selectbox(
                "Retrieval / Librarian Engine", 
                available_models, 
                index=get_idx(curr.get('model_retrieval', safe_default), available_models), 
                help="Handles 'Smart Search' to find relevant Lore/Facts. Speed (Flash) is recommended over raw power."
            )

        with c2:
            st.markdown("**🌍 World Mechanics**")
            tz = st.text_input("Timezone", value=curr.get('default_timezone', 'CST'))
            use_time = st.checkbox("Enable Time System", value=(curr.get('use_time_system', 'true') == 'true'))
            use_t = st.checkbox("Enable Multiverse / Timelines", value=(curr.get('use_timelines', 'true') == 'true'))

        # Dynamic Timeline Logic
        if use_t:
            st.divider()
            st.markdown("### 🌌 Multiverse Config")
            current_timelines = world_state.get("Timelines", [])
            df_timelines = pd.DataFrame(current_timelines)
            if df_timelines.empty:
                df_timelines = pd.DataFrame([{"Name": "Timeline Prime", "Description": "Main Reality"}])

            edited_timelines = st.data_editor(
                df_timelines,
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "Name": st.column_config.TextColumn("Timeline Name", required=True),
                    "Description": st.column_config.TextColumn("Context / Rules", width="large")
                },
                key="timeline_editor"
            )

        st.divider()
        if st.form_submit_button("Save Configuration", type="primary"):
            # Update Settings
            engine.update_story_setting(profile, 'use_time_system', str(use_time).lower())
            engine.update_story_setting(profile, 'default_timezone', tz)
            engine.update_story_setting(profile, 'use_timelines', str(use_t).lower())
            
            engine.update_story_setting(profile, 'model_scene', m_scene)
            engine.update_story_setting(profile, 'model_chat', m_chat)
            engine.update_story_setting(profile, 'model_reaction', m_react)
            engine.update_story_setting(profile, 'model_analysis', m_analysis)
            engine.update_story_setting(profile, 'model_retrieval', m_retrieval)
            
            # Update Timelines in JSON
            if use_t:
                world_state["Timelines"] = edited_timelines.to_dict(orient="records")
                engine.save_world_state(profile, world_state)
            
            st.success("Configuration Saved.")
            st.rerun()