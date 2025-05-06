import json
import requests
import time
import os
from datetime import datetime, timedelta

API_KEY = "b82d4ca2b04873bed15238bf21dc2de9b1b454565e49d3adfd7ec86d1c28985c"
KEYWORDS = ["Bitcoin", "Ethereum"]
START_DATE = "2023-11-22"
END_DATE = "2023-12-31"
SAVE_DIR = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\sentiment_data\google_search_data\real" 
os.makedirs(SAVE_DIR, exist_ok=True)  # Ensure directory exists

def generate_date_windows(start, end):
    """Generate 7-day sliding windows from start to end date"""
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    delta = end_date - start_date
    
    windows = []
    for i in range(delta.days - 6):
        window_start = start_date + timedelta(days=i)
        window_end = window_start + timedelta(days=6)
        windows.append((
            window_start.strftime("%Y-%m-%d"),
            window_end.strftime("%Y-%m-%d")
        ))
    return windows

def search_window(keyword, start, end):
    """Perform Google search for specific date window"""
    params = {
        "engine": "google",
        "q": f"{keyword} news",
        "api_key": API_KEY,
        "tbs": f"cdr:1,cd_min:{datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')},"
               f"cd_max:{datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')}",
        "num": 100
    }
    
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        return response.json()  # Return FULL API response
    except Exception as e:
        print(f"Error searching {keyword} {start}-{end}: {str(e)}")
        return None

def save_raw_data(keyword, start, end, data):
    """Save COMPLETE API response to file"""
    filename = f"{keyword.lower()}_{start}_{end}_FULL.json"
    save_path = os.path.join(SAVE_DIR, filename)
    
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved raw data to {save_path}")

def main():
    date_windows = generate_date_windows(START_DATE, END_DATE)
    
    for start, end in date_windows:
        for keyword in KEYWORDS:
            # Get full API response
            api_response = search_window(keyword, start, end)
            
            if api_response is not None:
                save_raw_data(keyword, start, end, api_response)
            
            time.sleep(2)  # Rate limiting

if __name__ == "__main__":
    main()