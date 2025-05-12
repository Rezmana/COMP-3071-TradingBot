import os
import json

def clean_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    organic_results = data.get('organic_results', [])

    cleaned_results = []
    for result in organic_results:
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        highlighted_words = result.get('snippet_highlighted_words', [])
        position = result.get('position', None)
        date = result.get('date', '')

        
        cleaned_results.append({
            'title': title,
            'snippet': snippet,
            'highlighted_words': highlighted_words,
            'position': position,
            'date': date
        })

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the cleaned results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_results, f, indent=4)

def batch_clean_jsons(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]  # Remove '.json'
            output_filename = f"{base_name}_cleaned.json"
            output_path = os.path.join(output_dir, output_filename)

            print(f"Processing: {filename} -> {output_filename}")
            clean_json_file(input_path, output_path)

BTCin_directory = r"Sentiment_Analysis\google scraping\raw data\bitcoin" 
BTCout_directory = r"Sentiment_Analysis\google scraping\cleaned data\bitcoin_cleaned" 
ETHin_directory = r"Sentiment_Analysis\google scraping\raw data\ethereum" 
ETHout_directory = r"Sentiment_Analysis\google scraping\cleaned data\ethereum_cleaned" 

batch_clean_jsons(BTCin_directory, BTCout_directory)
batch_clean_jsons(ETHin_directory, ETHout_directory)

