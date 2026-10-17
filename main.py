from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import pandas as pd

app = FastAPI()

# Cloud memory storage
latest_data = []

@app.post("/webhook")
async def receive_health_data(request: Request):
    """Parses the specific nested JSON payload from the Health Auto Export app."""
    payload = await request.json()
    
    # The app nests your data under payload["data"]["metrics"]
    data_block = payload.get("data", {})
    metrics = data_block.get("metrics", [])
    
    records_added = 0
    
    # Loop through each health metric category (e.g., heart_rate, step_count)
    for metric_group in metrics:
        metric_name = metric_group.get("name", "Unknown")
        
        # Loop through the actual timestamped readings inside that category
        for reading in metric_group.get("data", []):
            
            # The app sends either 'qty' (exact) or 'avg' (aggregated)
            val = reading.get("qty")
            if val is None:
                val = reading.get("avg")
                
            if val is not None:
                latest_data.append({
                    "metric": metric_name,
                    "timestamp": reading.get("date"),
                    "value": val
                })
                records_added += 1
                
    return {"status": "success", "records_parsed": records_added, "total_saved": len(latest_data)}

@app.get("/", response_class=HTMLResponse)
async def view_dashboard():
    """Serves the live dashboard HTML."""
    html_content = f"""
    <html>
        <body style="background:#0b0f19; color:#38bdf8; font-family:-apple-system, sans-serif; padding: 24px;">
            <h1>⚡ Cloud Dashboard Live</h1>
            <p style="font-size: 22px; color: #10b981;">Total sensor records synced: <b>{len(latest_data)}</b></p>
            <p style="color: #94a3b8; font-size: 14px;">(Refresh this page after tapping Sync in the iOS app)</p>
        </body>
    </html>
    """
    return html_content
