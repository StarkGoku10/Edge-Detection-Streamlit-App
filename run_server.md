# How to Run the Edge Detection Server

## Option 1: Run from the backend directory (Recommended)
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Option 2: Run from the project root directory
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Option 3: Run with Python directly (Alternative)
```bash
cd backend
python main.py
```

## Troubleshooting

### If you see "Frontend not found" error:
1. Check that you're running from the correct directory
2. Verify the frontend directory exists and contains index.html
3. Check the console output for path debugging information

### If the frontend loads but file upload fails:
1. Open browser console (F12)
2. Look for JavaScript errors during file selection
3. Check network tab for failed API requests

### Current Project Structure Expected:
```
Edge-Detection-Streamlit-App/
├── backend/
│   ├── main.py
│   └── edge_detection.py
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
└── requirements.txt
```

## Testing the Server
1. Start the server using one of the commands above
2. Open http://localhost:8000 in your browser
3. You should see the Edge Detection Studio interface
4. Check the server console for path debugging information 