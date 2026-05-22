=== CHRONOS STORY DIRECTOR — STARTUP CHEATSHEET ===

--- ENVIRONMENT ---
source venv/Scripts/activate          # Windows Git Bash / Linux / Mac
# OR
.\venv\Scripts\Activate.ps1           # Windows PowerShell

--- BACKEND (run from repo root) ---
uvicorn src.api:app --reload

--- FRONTEND (run in a second terminal) ---
cd frontend
npm run dev

--- GIT ---
git pull --rebase origin main         # pull latest before working
git add .                             # stage all changes
git commit -m "your message"          # commit
git push                              # push to GitHub

--- NOTES ---
- Backend runs on http://localhost:8000
- Frontend runs on http://localhost:5173
- Always activate venv before running uvicorn
- Run backend and frontend in separate terminals simultaneously