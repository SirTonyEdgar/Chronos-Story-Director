# 🕰️ Chronos Story Director

**Chronos Story Director** is a sophisticated RAG-based storytelling engine designed to orchestrate LLMs (Gemini, GPT-4, Claude) for long-form narrative generation. It tracks world state, manages complex lore, and provides a "War Room" for strategic simulations.

![Version](https://img.shields.io/badge/version-15.0-blue)
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

<img width="1047" height="850" alt="image" src="https://github.com/user-attachments/assets/9238da8a-993a-42dd-9e09-14381cb70e3b" />
<img width="1063" height="860" alt="image" src="https://github.com/user-attachments/assets/7612972a-9286-4f95-a55c-e6a22938466f" />
<img width="1046" height="809" alt="image" src="https://github.com/user-attachments/assets/a5285f59-c2c0-4177-a7d6-1668ab74cc49" />
<img width="1057" height="811" alt="image" src="https://github.com/user-attachments/assets/56b86856-18a7-4301-a548-c67196e0c54f" />
<img width="997" height="894" alt="image" src="https://github.com/user-attachments/assets/106d286a-32fb-4efa-bda0-49836fdb1320" />
<img width="1564" height="887" alt="image" src="https://github.com/user-attachments/assets/3c1f3534-c294-448c-acec-123d394d34ae" />
<img width="1400" height="899" alt="image" src="https://github.com/user-attachments/assets/2af14518-1bc4-45a5-bb90-19761c75ff81" />
<img width="1468" height="909" alt="image" src="https://github.com/user-attachments/assets/eb3e78b0-b933-47d9-94e5-bd12c1b3a671" />
<img width="1877" height="925" alt="image" src="https://github.com/user-attachments/assets/68e903a5-cb32-450f-86f7-0bb31773dd23" />
<img width="1874" height="927" alt="image" src="https://github.com/user-attachments/assets/4a91cdff-e2c2-45d2-bb93-84acb8b83e79" />
<img width="1887" height="919" alt="image" src="https://github.com/user-attachments/assets/f734c298-9ff2-4b5c-9a2a-84a4b55b7843" />
<img width="1864" height="804" alt="image" src="https://github.com/user-attachments/assets/7969369f-797b-488f-8b58-f371d1f34a1b" />

## 🛠️ Installation

### 1. Clone the Repository
- ```git clone https://github.com/SirTonyEdgar/Chronos-Story-Director.git```
- ```cd Chronos-Story-Director```

### 2. Backend Setup (Python)
Create a virtual environment:
```python -m venv venv```

### Activate virtual environment:
### Windows:
```venv\Scripts\activate```
### Mac/Linux:
```source venv/bin/activate```

### Install dependencies
```pip install -r requirements.txt```

### 3. Setup API Keys
Create a file named .env in the root directory and add your API keys:
- ```GOOGLE_API_KEY=your_gemini_key_here```
- ```OPENAI_API_KEY=your_openai_key_here```
- ```ANTHROPIC_API_KEY=your_claude_key```

### 4. Frontend Setup (React/Node)
Open a new terminal window in the project root:
- ```cd frontend```
- ```npm install```

## 🚀 Running Chronos
You will need two terminal windows running simultaneously to power the engine.
### Terminal 1: Start the Backend API (Ensure your python virtual environment is active):
- ```uvicorn src.api:app --reload --port 8000```

### Terminal 2: Start the React UI
- ```cd frontend```
- ```npm run dev```

Once both are running, open your browser and navigate to http://localhost:5173 (or the port provided by Vite/React) to access the Story Director.

## 📄 License
This project is licensed under the MIT License - see the source header for details.
