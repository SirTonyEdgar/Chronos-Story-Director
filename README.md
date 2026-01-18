# 🕰️ Chronos Story Director

**Chronos Story Director** is a sophisticated RAG-based storytelling engine designed to orchestrate LLMs (Gemini, GPT-4) for long-form narrative generation. It tracks world state, manages complex lore, and provides a "War Room" for strategic simulations.

![Version](https://img.shields.io/badge/version-11.7-blue)
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

<img width="1440" height="639" alt="image" src="https://github.com/user-attachments/assets/0f6f2f2b-c830-4db1-803c-623d9fff5a89" />
<img width="1429" height="876" alt="image" src="https://github.com/user-attachments/assets/558919bf-f7d9-4334-ab6a-63371f304dcb" />
<img width="1444" height="456" alt="image" src="https://github.com/user-attachments/assets/fc7e0151-1a0f-43d2-be98-c30b66a59d38" />
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
