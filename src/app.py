"""
Chronos Story Director - User Interface
=======================================
The Streamlit-based frontend application for the Chronos engine.
This module handles user interaction, visualizes world state via interactive
graphs, and orchestrates the narrative generation workflow.

Copyright (c) 2025 SirTonyEdgar
Licensed under the MIT License.
"""

import os
import io
import json
import glob
import datetime
import math
import uuid
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from streamlit_agraph import agraph, Node, Edge, Config

# Internal Engine Import
import backend as engine

# --- GLOBAL CONFIGURATION & ASSETS ---
st.set_page_config(
    page_title="Chronos Director", 
    page_icon="🕰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Centralized Icon Repository to ensure consistency across Status and Network tabs
AVATAR_MANIFEST = {
    "Male": "https://img.icons8.com/plasticine/100/user.png",
    "Female": "https://img.icons8.com/plasticine/100/user-female.png",
    "Neutral": "https://img.icons8.com/plasticine/100/test-account.png", 
    "King": "https://img.icons8.com/plasticine/100/crown.png",          
    "Queen": "https://img.icons8.com/plasticine/100/jewelry.png",      
    "Wizard": "https://img.icons8.com/plasticine/100/wizard.png",
    "Cyborg": "https://img.icons8.com/plasticine/100/bot.png",
    "Cat": "https://img.icons8.com/plasticine/100/cat.png"
}

RELATIONSHIP_ICONS = {
    "Ally": "https://img.icons8.com/plasticine/100/handshake.png",
    "Family": "https://img.icons8.com/plasticine/100/family.png",
    "Enemy": "https://img.icons8.com/plasticine/100/skull.png",
    "Rival": "https://img.icons8.com/plasticine/100/fencing.png",
    "Love": "https://img.icons8.com/plasticine/100/like--v1.png",
    "Mentor": "https://img.icons8.com/plasticine/100/wizard.png",
    "Vassal": "https://img.icons8.com/plasticine/100/kneeling.png",
    "Target": "https://img.icons8.com/plasticine/100/goal.png",
    "Resource": "https://img.icons8.com/plasticine/100/money-bag.png",
    "Unknown": "https://img.icons8.com/plasticine/100/question-mark.png"
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

# Sidebar Navigation
with st.sidebar:
    st.title("🕰️ Chronos")
    st.caption(f"Profile: **{profile}**")
    
    if st.button("🔄 Switch Profile"):
        st.session_state.clear() 
        st.query_params.clear()
        st.rerun()
    
    st.divider()
    
    nav_options = [
        "🎬 Scene Creator", 
        "🚧 Projects", 
        "⚔️ War Room", 
        "🕸️ Network Map",  # Dedicated Page
        "📊 Status & Assets", 
        "📚 Compiler", 
        "🗄️ Knowledge Base", 
        "🗣️ Reaction Tool", 
        "🧠 Co-Author Chat", 
        "⚙️ Settings"
    ]
    page = st.radio("Navigation", nav_options, label_visibility="collapsed")
    
    st.divider()
    st.caption("v10.6 - GitHub Release")

# ==========================================
# MODULE: SCENE CREATOR
# ==========================================
if page == "🎬 Scene Creator":
    st.header("🎬 Scene Creator")
    tab_write, tab_read, tab_edit, tab_manage = st.tabs(["✍️ Write", "📖 Read", "✏️ Edit", "🗑️ Manage"])
    
    with tab_write:
        # Load constraints
        settings = engine.get_story_settings(profile)
        use_time = settings.get('use_time_system', 'true').lower() == 'true'

        year = 0
        date_str = ""
        time_str = ""

        # Time System Controls
        if use_time:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1: 
                year = st.number_input("Year", value=None, step=1, placeholder="1984", format="%d", key="sc_year")
            with c2: 
                date_str = st.text_input("Date", placeholder="e.g. March 6th", key="sc_date")
            with c3: 
                time_str = st.text_input("Time", placeholder="Auto-Detect If Left Empty", key="sc_time")
        else:
            st.caption("⏳ Time System is Disabled.")

        # Briefing Inputs
        title = st.text_input("Scene Title", placeholder="Required", key="sc_title")
        all_files = ["Auto (Last 3 Scenes)"] + engine.get_all_files_list(profile)
        context_files = st.multiselect("Transition From:", all_files, default=[], key="sc_context")
        brief = st.text_area("Scene Brief", height=150, key="sc_brief", placeholder="Describe the action...")
        
        # Privacy Control (Fog of War)
        auto_privacy = st.checkbox("🕵️ Enable Fog of War (Auto-Tag Private Scenes)", value=False, help="If enabled, the AI will wrap private conversations in [[PRIVATE]] tags. Required for the Reaction Tool's 'Public Knowledge' mode.")

        if st.button("Generate Scene", type="primary", use_container_width=True, key="btn_gen_scene"):
            if not title or not brief: 
                st.error("Title and Brief required.")
            else:
                with st.spinner("Drafting Narrative..."):
                    text, path = engine.generate_scene(
                        profile, year, date_str, time_str, title, brief, 
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
        if files:
            sel = st.selectbox("Edit File:", files, key="sc_edit_select")
            cont = engine.read_file_content(profile, sel)
            new_c = st.text_area("Editor", value=cont, height=600, key="sc_editor_area")
            if st.button("Save Changes", key="btn_save_edit"):
                engine.save_edited_scene(profile, sel, new_c)
                st.success("File Updated!")

    with tab_manage:
        paths = engine.get_paths(profile)
        files = [os.path.basename(f) for f in glob.glob(os.path.join(paths['output'], "*.txt"))]
        if files:
            sel = st.selectbox("Delete File:", files, key="sc_del_select")
            if st.button("Permanently Delete", type="secondary", key="btn_del_file"):
                engine.delete_specific_scene(profile, sel)
                st.rerun()

# ==========================================
# MODULE: PROJECTS
# ==========================================
elif page == "🚧 Projects":
    st.header("🚧 Project & Tech Tracker")
    state = engine.get_world_state(profile)
    projects = state.get("Projects", [])
    
    tab_active, tab_new = st.tabs(["Active Projects", "Start New Project"])
    
    with tab_active:
        if not projects: 
            st.info("No active projects tracked.")
        else:
            for i, proj in enumerate(projects):
                with st.expander(f"{proj['Name']} ({proj['Progress']}%)"):
                    st.write(f"**Goal:** {proj['Description']}")
                    new_specs = st.text_area(f"Specs", value=proj['Features_Specs'], key=f"spec_{i}", height=100)
                    new_prog = st.slider("Progress", 0, 100, proj['Progress'], key=f"prog_{i}")
                    
                    c1, c2 = st.columns([1, 1])
                    if c1.button("Update Status", key=f"upd_{i}"):
                        engine.update_project(profile, i, new_prog, new_specs)
                        st.success("Project Updated")
                        st.rerun()
                    
                    with st.popover("✅ Archive / Complete"):
                        st.markdown(f"**Finalize: {proj['Name']}**")
                        default_history = f"Completed. Specs: {new_specs}"
                        final_history = st.text_area("Historical Record", value=default_history, height=150)
                        if st.button("Confirm Completion", key=f"arch_{i}"):
                            success, msg = engine.complete_project(profile, i, final_history)
                            if success:
                                st.balloons()
                                st.success(msg)
                                st.rerun()
    
    with tab_new:
        with st.form("new_proj"):
            n_name = st.text_input("Project Name")
            n_desc = st.text_input("Objective")
            n_specs = st.text_area("Initial Specifications")
            if st.form_submit_button("Launch Project"):
                engine.add_project(profile, n_name, n_desc, n_specs)
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
# MODULE: NETWORK MAP (DEDICATED)
# ==========================================
elif page == "🕸️ Network Map":
    st.header("🕸️ Empire Network Map")
    
    # 1. Load State
    if 'dashboard_state' not in st.session_state:
        st.session_state['dashboard_state'] = engine.get_world_state(profile)
    current_state = st.session_state['dashboard_state']

    # 2. Controls & Interaction Lock
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
    font_style = {'color': 'white', 'strokeWidth': 4, 'strokeColor': 'black', 'size': 18}

    # 3. Build Nodes
    p_data = current_state.get("Protagonist Status", {})
    p_name = p_data.get("Name", "Protagonist")
    p_icon_key = p_data.get("Icon", "Male")
    
    # Resolve Icon from Global Manifest
    p_icon_url = AVATAR_MANIFEST.get(p_icon_key, AVATAR_MANIFEST["Male"])
    
    # Central Node (Pinned)
    nodes.append(Node(
        id="MAIN", label=p_name, size=50, shape="circularImage", 
        image=p_icon_url,
        font=font_style, x=0, y=0, fixed=True
    ))
    
    # Neighbor Nodes
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

    for ally in allies:
        name = ally.get("Name", "Unknown")
        # Default to 'Ally' handshake if not found
        icon_url = FULL_ICON_MAP.get(ally.get("Icon", "Ally"), RELATIONSHIP_ICONS["Ally"])
        
        lx, ly = get_coords(current_idx, total, radius)
        current_idx += 1
        nodes.append(Node(id=name, label=name, size=30, shape="circularImage", image=icon_url, font=font_style, x=lx, y=ly, fixed=True))
        edges.append(Edge(source="MAIN", target=name, label=ally.get("Relation", "Ally"), color="#4CAF50"))

    for asset in assets:
        name = asset.get("Asset", "Unknown")
        lx, ly = get_coords(current_idx, total, radius)
        current_idx += 1
        nodes.append(Node(id=name, label=name, size=25, shape="circularImage", image=RELATIONSHIP_ICONS["Resource"], font=font_style, x=lx, y=ly, fixed=True))
        edges.append(Edge(source="MAIN", target=name, label=asset.get("Type", "Resource"), color="#FFC107"))

    # 4. Render
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
# MODULE: STATUS & ASSETS
# ==========================================
elif page == "📊 Status & Assets":
    st.header("📊 World State Tracker")
    
    if 'dashboard_state' not in st.session_state:
        st.session_state['dashboard_state'] = engine.get_world_state(profile)
    
    # Process batch analysis callbacks
    if 'temp_state' in st.session_state:
        st.session_state['dashboard_state'] = st.session_state.pop('temp_state')
        st.toast("State updated from AI Analysis!")

    current_state = st.session_state['dashboard_state']
    
    # --- AI ANALYSIS TOOLS ---
    with st.expander("🤖 AI Batch Analysis Tools", expanded=False):
        st.caption("Select scenes to let AI auto-extract changes to allies, assets, and skills.")
        paths = engine.get_paths(profile)
        scene_files = [os.path.basename(f) for f in glob.glob(os.path.join(paths['output'], "*.txt"))]
        scene_files.sort(key=lambda x: os.path.getmtime(os.path.join(paths['output'], x)), reverse=True)
        
        if scene_files:
            target_scenes = st.multiselect("Select Scenes:", scene_files, default=[])
            if st.button("Analyze & Update State", width="stretch"):
                if target_scenes:
                    with st.spinner("Processing Context..."):
                        combined = ""
                        for fname in target_scenes:
                            combined += f"\n=== {fname} ===\n{engine.read_file_content(profile, fname)}\n"
                        st.session_state['temp_state'] = engine.analyze_state_changes(profile, combined)
                        st.rerun()

    st.divider()
    st.subheader("🏰 Empire Dashboard")
    
    tab_main, tab_allies, tab_assets, tab_skills, tab_raw = st.tabs(
        ["👤 Protagonist", "🤝 Relations", "💰 Assets", "⚡ Skills", "📝 JSON"]
    )
    
    # --- TAB: PROTAGONIST IDENTITY ---
    with tab_main:
        p_data = current_state.get("Protagonist Status", {})
        if not isinstance(p_data, dict): p_data = {}
        
        # Sync with DB Settings
        sys_settings = engine.get_story_settings(profile)
        
        st.info("🎭 **Identity Management:** The AI uses 'True Identity' for memory, but writes the story using 'Current Name'.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📖 Narrative Persona")
            p_data["Name"] = st.text_input("Current Narrative Name", value=p_data.get("Name", "Unknown"), help="The name used in prose (e.g. 'Anthony walked...').")
            p_data["Aliases"] = st.text_area("Known Aliases / Nicknames", value=p_data.get("Aliases", ""), help="Comma-separated list (e.g. 'Tony, The Kid').", height=68)
            
            # Icon Selector (Uses Global Manifest)
            current_icon = p_data.get("Icon", "Male")
            if current_icon not in AVATAR_MANIFEST: current_icon = "Male"
            p_data["Icon"] = st.selectbox("Map Avatar", options=list(AVATAR_MANIFEST.keys()), index=list(AVATAR_MANIFEST.keys()).index(current_icon), key="p_avatar")

        with c2:
            st.subheader("🧠 Deep Memory")
            true_name = st.text_input("True Identity (Context)", value=sys_settings.get('protagonist', ''), help="Original identity used for AI context grounding.")
            p_data["Age"] = st.text_input("Age/Status", value=str(p_data.get("Age", "Unknown")))
            p_data["Goal"] = st.text_area("Current Goal", value=p_data.get("Current Goal", ""), height=100)

        # Avatar Preview
        st.caption("Avatar Preview:")
        st.image(AVATAR_MANIFEST[p_data["Icon"]], width=60)
        
        current_state["Protagonist Status"] = p_data
        
        if st.button("💾 Update Identity", type="primary"):
            engine.save_world_state(profile, current_state)
            engine.update_story_setting(profile, 'protagonist', true_name)
            st.toast("Identity Updated.", icon="✅")
            st.rerun()

    # --- TAB: RELATIONS ---
    with tab_allies:
        target_list = current_state.get("Allies", [])
        df_allies = pd.DataFrame(target_list)
        
        if df_allies.empty:
            df_allies = pd.DataFrame(columns=["Name", "Relation", "Loyalty", "Notes", "Icon"])
        
        if "Icon" not in df_allies.columns:
            df_allies["Icon"] = "Ally"

        # Legend
        with st.expander("🖼️ Icon Reference"):
            st.caption("Use these keys in the **Map Icon** column.")
            cols = st.columns(9)
            i = 0
            for name, url in RELATIONSHIP_ICONS.items():
                with cols[i % 9]:
                    st.image(url, width=40)
                    st.caption(name)
                i += 1

        edited_allies_df = st.data_editor(
            df_allies,
            num_rows="dynamic", 
            width="stretch",
            column_config={
                "Name": st.column_config.TextColumn("Name", required=True),
                "Relation": st.column_config.TextColumn("Relation", default="Ally"),
                "Loyalty": st.column_config.NumberColumn("Loyalty %", min_value=0, max_value=100, default=50),
                "Icon": st.column_config.SelectboxColumn(
                    "Map Icon",
                    options=list(RELATIONSHIP_ICONS.keys()),
                    required=True,
                    default="Ally",
                    help="Select an icon type."
                ),
                "Notes": st.column_config.TextColumn("Notes")
            },
            key="editor_allies"
        )
        current_state["Allies"] = edited_allies_df.to_dict(orient="records")

    # --- TAB: ASSETS ---
    with tab_assets:
        assets_list = current_state.get("Assets", [])
        df_assets = pd.DataFrame(assets_list)
        if df_assets.empty:
            df_assets = pd.DataFrame(columns=["Asset", "Type", "Status", "Value"])
        
        edited_assets_df = st.data_editor(
            df_assets,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Asset": st.column_config.TextColumn("Asset Name", required=True),
                "Type": st.column_config.TextColumn("Type", default="Financial"),
                "Status": st.column_config.TextColumn("Status", default="Active"),
                "Value": st.column_config.TextColumn("Value/Power", default="Unknown")
            },
            key="editor_assets"
        )
        current_state["Assets"] = edited_assets_df.to_dict(orient="records")

    # --- TAB: SKILLS ---
    with tab_skills:
        skills = current_state.get("Skills", [])
        # Normalization for list-of-strings vs list-of-dicts
        skill_dicts = [{"Skill": s} for s in skills] if skills and isinstance(skills[0], str) else (skills or [])
        
        edited_skills = st.data_editor(
            skill_dicts,
            num_rows="dynamic",
            width="stretch",
            column_config={"Skill": st.column_config.TextColumn("Skill Name", required=True)},
            key="editor_skills"
        )
        current_state["Skills"] = [d["Skill"] for d in edited_skills if d.get("Skill")]

    # --- TAB: RAW JSON ---
    with tab_raw:
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
# MODULE: COMPILER
# ==========================================
elif page == "📚 Compiler":
    st.header("📚 Manuscript Compiler")
    all_scenes = engine.get_all_files_list(profile)
    selected = st.multiselect("Select Scenes to compile (Ordered):", all_scenes)
    
    if selected and st.button("Compile Manuscript"):
        text = engine.compile_manuscript(profile, selected)
        st.download_button("Download as Markdown", text, "Book.md")
        st.download_button("Download as Text", text, "Book.txt")

# ==========================================
# MODULE: KNOWLEDGE BASE
# ==========================================
elif page == "🗄️ Knowledge Base":
    st.header("🗄️ Knowledge Base")
    tab_lore, tab_rules, tab_plan, tab_fact, tab_spoiler = st.tabs(
        ["📜 Lore", "📏 Rules", "🗺️ Plans", "📌 Facts", "🚫 Spoilers"]
    )

    def render_editor(doc_type, label):
        """Standardized editor component for Knowledge Base fragments."""
        frags = engine.get_fragments(profile, doc_type)
        if frags:
            for fid, fname, cont, _ in frags:
                with st.expander(f"[{fid}] {fname}"):
                    with st.form(f"e_{fid}"):
                        txt = st.text_area("Content", value=cont, height=200)
                        c1, c2 = st.columns([0.2, 0.8])
                        if c1.form_submit_button("Update"): 
                            engine.update_fragment(profile, fid, txt)
                            st.rerun()
                        if c2.form_submit_button("Delete"): 
                            engine.delete_fragment(profile, fid)
                            st.rerun()
        st.divider()
        
        # Uploader Section
        up = st.file_uploader(f"Upload {label}", type=["txt", "pdf"], key=f"u_{doc_type}")
        if up and st.button(f"Import {up.name}", key=f"i_{doc_type}"):
            content = None
            if up.type == "text/plain": 
                content = up.getvalue().decode("utf-8")
            elif up.type == "application/pdf":
                try:
                    pdf = PdfReader(io.BytesIO(up.getvalue()))
                    content = "".join([page.extract_text() + "\n" for page in pdf.pages])
                except Exception as e:
                    st.error(f"PDF Error: {e}")
            
            if content:
                engine.add_fragment(profile, up.name, content, doc_type)
                st.success(f"Imported {up.name}")
                st.rerun()
        
        # Manual Entry Section
        with st.form(f"a_{doc_type}"):
            n = st.text_input("Title")
            c = st.text_area("Content")
            if st.form_submit_button("Add New"): 
                engine.add_fragment(profile, n, c, doc_type)
                st.rerun()

    with tab_lore: render_editor("Lore", "Lore")
    
    with tab_rules: 
        st.info("📜 **World Laws:** Upload RPG Rulebooks or Physics constraints. The AI treats these as strict instructions.")
        render_editor("Rulebook", "Rules")
    
    with tab_plan: render_editor("Plan", "Plan")
    
    with tab_fact:
        frags = engine.get_fragments(profile, "Fact")
        for fid, fname, cont, _ in frags:
            c1, c2 = st.columns([0.9, 0.1])
            c1.info(cont)
            if c2.button("X", key=f"df_{fid}"): 
                engine.delete_fragment(profile, fid)
                st.rerun()
        with st.form("af"):
            nf = st.text_input("New Fact")
            if st.form_submit_button("Add"): 
                engine.add_fragment(profile, "UI", nf, "Fact")
                st.rerun()
                
    with tab_spoiler:
        frags = engine.get_fragments(profile, "Spoiler")
        for fid, fname, cont, _ in frags:
            c1, c2 = st.columns([0.9, 0.1])
            c1.error(cont)
            if c2.button("X", key=f"ds_{fid}"): 
                engine.delete_fragment(profile, fid)
                st.rerun()
        with st.form("as"):
            ns = st.text_input("Banned Term")
            if st.form_submit_button("Ban"): 
                engine.add_fragment(profile, "UI", ns, "Spoiler")
                st.rerun()

# ==========================================
# MODULE: REACTION TOOL (Genre-Agnostic)
# ==========================================
elif page == "🗣️ Reaction Tool":
    st.header("🗣️ Faction Reaction Engine")
    files = engine.get_all_files_list(profile)
    
    # Template Library for Sub-Options
    REACTION_TEMPLATES = {
        "🌐 Modern Internet": [
            "Anonymous Imageboard (e.g. 4chan)",
            "Social Media Feed (e.g. Twitter/X)",
            "Video Comment Section (e.g. YouTube)",
            "Niche Hobby Forum (e.g. Reddit/NMA)",
            "Group Chat Leak (e.g. Discord/WhatsApp)"
        ],
        "📜 Fantasy / Medieval": [
            "Tavern Rumors / Gossip",
            "Town Crier Announcement",
            "Royal Court Whispers",
            "Adventurer's Guild Quest Board",
            "Magic Message / Telepathy"
        ],
        "🚀 Sci-Fi / Cyberpunk": [
            "Holonet Data Stream",
            "Corporate News Flash",
            "Hacker Encrypted Channel",
            "Ship-to-Ship Comms",
            "Android Memory Log"
        ],
        "📺 20th Century / Retro": [
            "Newspaper Front Page",
            "Radio Broadcast Transcript",
            "Telegraph / Cable",
            "Underground Pamphlet"
        ],
        "✨ Custom / Other": ["Manual Input"]
    }

    if files:
        # Context Selection
        c1, c2 = st.columns([1, 1])
        with c1:
            target_scene = st.selectbox("Select Context Scene:", files)
        with c2:
            target_faction = st.text_input("Target Faction/Character:", placeholder="e.g. The peasantry, /pol/, The Galactic Senate")

        # Format Selection
        c3, c4 = st.columns([1, 1])
        with c3:
            selected_category = st.selectbox("Era / Tech Level", list(REACTION_TEMPLATES.keys()))
        
        with c4:
            available_formats = REACTION_TEMPLATES[selected_category]
            selected_format = st.selectbox("Format / Medium", available_formats)

        # Custom Format Override
        final_style_instruction = selected_format
        if selected_format == "Manual Input":
            final_style_instruction = st.text_input("Describe Custom Format:", placeholder="e.g. Telepathic dream sequence")

        # Fog of War Control
        st.caption("---")
        is_public = st.checkbox("👁️ Public Knowledge Only", value=True, help="If checked, the AI ignores secrets/internal thoughts and only reacts to what is visible.")
        
        st.divider()
        
        if st.button("Simulate Reaction", type="primary"):
            if not target_faction:
                st.error("Please specify a Faction.")
            else:
                full_style_prompt = f"{selected_category} -> {final_style_instruction}"
                
                with st.spinner(f"Simulating ({final_style_instruction})..."):
                    success, res = engine.generate_reaction_for_scene(
                        profile, target_scene, target_faction, 
                        public_only=is_public, 
                        format_style=full_style_prompt
                    )
                    
                    if success: 
                        st.success("Reaction Generated & Saved!")
                        st.markdown(f"### Output Preview")
                        st.write(res)
                    else: 
                        st.error(res)

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
# MODULE: SETTINGS
# ==========================================
elif page == "⚙️ Settings":
    st.header("⚙️ System Configuration")
    
    curr = engine.get_story_settings(profile)
    world_state = engine.get_world_state(profile)
    available_models = get_cached_models()
    
    with st.form("global_settings_form"):
        st.subheader("🌍 World Mechanics")
        c1, c2 = st.columns(2)
        with c1:
            tz = st.text_input("Timezone", value=curr.get('default_timezone', 'CST'))
            use_time = st.checkbox("Enable Time System", value=(curr.get('use_time_system', 'true') == 'true'))
            use_t = st.checkbox("Enable Multiverse / Timelines", value=(curr.get('use_timelines', 'true') == 'true'))
        
        with c2:
            st.markdown("**🧠 AI Model Routing**")
            safe_default = available_models[0] if available_models else "No Models Detected"
            def get_idx(val, opts): return opts.index(val) if val in opts else 0
            
            m_scene = st.selectbox("Scene Writer", available_models, index=get_idx(curr.get('model_scene', safe_default), available_models))
            m_chat = st.selectbox("Co-Author", available_models, index=get_idx(curr.get('model_chat', safe_default), available_models))
            m_react = st.selectbox("Reaction Engine", available_models, index=get_idx(curr.get('model_reaction', safe_default), available_models))

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
            
            # Update Timelines in JSON
            if use_t:
                world_state["Timelines"] = edited_timelines.to_dict(orient="records")
                engine.save_world_state(profile, world_state)
            
            st.success("Configuration Saved.")
            st.rerun()