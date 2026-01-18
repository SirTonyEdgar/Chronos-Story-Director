# 🕰️ Chronos Story Director

**Chronos Story Director** is a sophisticated RAG-based storytelling engine designed to orchestrate LLMs (Gemini, GPT-4) for long-form narrative generation. It tracks world state, manages complex lore, and provides a "War Room" for strategic simulations.

![Version](https://img.shields.io/badge/version-11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- **🎬 Scene Creator:** Auto-generate prose with consistent history tracking.
- **🌌 Multiverse Support:** Manage multiple timelines and parallel realities.
- **⚔️ War Room:** Monte Carlo-style strategy simulator for plot decisions.
- **🕸️ Interactive Network Map:** Dedicated full-screen visualization of character relationships.
- **📊 World State Tracker:** JSON-based tracking of Assets, Allies, and Skills.
- **🧠 Co-Author Chat:** RAG-aware chatbot that knows your entire story bible.
- **🔧 Normie-Friendly UI:** Built with Streamlit for a clean, visual experience.
- **🕵️ Fog of War:** Redact private scenes so the "Public Reaction" AI doesn't know your secrets.
- **🎭 Identity Manager:** Handle aliases, secret identities, and changing narrative personas.

<img width="1872" height="921" alt="Scene_Creator_Chronos_Story_Engine" src="https://github.com/user-attachments/assets/dd577e8b-cb7d-47d0-b201-5daf48747d01" />
<img width="1872" height="923" alt="War_Room_Chronos_Story_Engine" src="https://github.com/user-attachments/assets/ef2517c2-fc0f-4b4d-ae08-0ed03003f5c0" />

## 🛠️ Installation

### 1. Clone the Repository
- ```git clone https://github.com/SirTonyEdgar/Chronos-Story-Director.git```
- ```cd Chronos-Story-Director```

### 2. Install Dependencies
```pip install -r requirements.txt```

### 3. Setup API Keys
Create a file named .env in the root directory and add your API keys:
- ```GOOGLE_API_KEY=your_gemini_key_here```
- ```OPENAI_API_KEY=your_openai_key_here```  # Optional

## 🚀 Usage
Run the interface from your terminal:
```streamlit run src/app.py```

## 📂 Project Structure
- ```src/app.py```: The Streamlit frontend interface.
- ```src/backend.py```: The core LangGraph engine and logic.
- ```src/database_manager.py```: CLI tool for bulk ingesting lore/PDFs.
- ```profiles/```: Stores your local story data (ignored by git).

## 📄 License
This project is licensed under the MIT License - see the source header for details.
