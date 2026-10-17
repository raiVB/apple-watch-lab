from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import pandas as pd
from datetime import datetime

app = FastAPI()

# In-memory storage for free tier (Render ephemeral disk wipes on restart)
# For a permanent free database, you would eventually connect SQLite or PostgreSQL
latest_data = []

@app.post("/webhook")
async def receive_health_data(request: Request):
    """Receives JSON data pushed from Apple Shortcuts."""
    payload = await request.json()
    
    for sample in payload.get("data", []):
        latest_data.append({
            "metric": sample.get("type"),
            "timestamp": sample.get("startDate"),
            "value": sample.get("value")
        })
    
    return {"status": "success", "records_logged": len(latest_data)}

@app.get("/", response_class=HTMLResponse)
async def view_dashboard():
    """Serves the dashboard HTML."""
    # In a full deployment, this is where you would call your build_dashboard() 
    # function to generate the HTML string based on the data.
    html_content = f"""
    <html>
        <body style="background:#0b0f19; color:#38bdf8; font-family:sans-serif; padding: 20px;">
            <h1>⚡ Cloud Dashboard Live</h1>
            <p>Total sensor records synced: {len(latest_data)}</p>
        </body>
    </html>
    """
    return html_content
