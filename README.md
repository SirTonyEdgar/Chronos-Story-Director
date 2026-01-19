# 🕰️ Chronos Story Director

**Chronos Story Director** is a sophisticated RAG-based storytelling engine designed to orchestrate LLMs (Gemini, GPT-4) for long-form narrative generation. It tracks world state, manages complex lore, and provides a "War Room" for strategic simulations.

![Version](https://img.shields.io/badge/version-12.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- **🎬 Context-Aware Scene Creator:** Auto-generate prose that respects your lore, rules, and previous chapters.
- **🌌 World Simulation Engine:** Track abstract forces (e.g., "Federal Heat," "Timeline Drift") with rule-based consequences.
- **⚔️ War Room:** Run Monte Carlo-style strategic simulations to predict the risks of your plot decisions.
- **📚 Manuscript Publisher:** Compile your scenes into print-ready PDF or EPUB books with one click.
- **🕸️ Interactive Network Map:** A live, physics-based graph visualization of character relationships.
- **📊 Dynamic State Tracker:** JSON-based persistence for Assets, Allies, Skills, and Projects.
- **🧠 RAG Co-Author:** A chat assistant that knows your entire story bible, lore, and secret plans.
- **🕵️ Fog of War:** "Redacted" text handling allows you to simulate Public Reactions to your secret moves.
- **🌌 Multiverse Support:** Manage branching timelines and parallel realities without database corruption.
- **🎭 Identity Manager:** Handle aliases, secret identities, and changing narrative personas.

<img width="1440" height="639" alt="image" src="https://github.com/user-attachments/assets/0f6f2f2b-c830-4db1-803c-623d9fff5a89" />
<img width="1449" height="822" alt="image" src="https://github.com/user-attachments/assets/8f3c158c-bf9d-4001-b2cb-d18b0101e603" />
<img width="1432" height="849" alt="image" src="https://github.com/user-attachments/assets/eb59a465-024b-4464-a098-8f0901e84552" />
<img width="1431" height="875" alt="image" src="https://github.com/user-attachments/assets/c7e48085-e1bf-48e7-b9a3-2a8b79985b63" />
<img width="1452" height="494" alt="image" src="https://github.com/user-attachments/assets/01f85c8f-5cf9-417d-8b2f-2aff4d14d551" />
<img width="1436" height="845" alt="image" src="https://github.com/user-attachments/assets/999f8932-04ac-4c24-afba-ffe62ceece8a" />
<img width="1446" height="761" alt="image" src="https://github.com/user-attachments/assets/379a30b2-369d-46f8-9139-f0bb8031115b" />
<img width="1437" height="865" alt="image" src="https://github.com/user-attachments/assets/33f5f3a0-00e4-4368-9ad8-12cfa9dde0b5" />

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
