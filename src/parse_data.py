import os
import json
import logging
import argparse
import shutil
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup

from api_clean import llm_api_clean

# --- Setup Functions ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse PDF documents based on an index file and clean content using an API.")
    parser.add_argument('--title', type=str, required=True, help="Primary directory name for data and output.")
    parser.add_argument('--subtitle', type=str, required=True, help="Secondary directory name for data and output.")
    parser.add_argument('--logging', type=str, required=True, help="Path to the log file.")
    return parser.parse_args()

def setup_logger(log_file, logger_name):
    """Configures and returns a logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

# --- File I/O and Parsing Functions ---

def read_index_file(index_path, logger):
    """Reads an index.htm file and returns a list of file metadata."""
    logger.info(f"Reading index file from: {index_path}")
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
    except FileNotFoundError:
        logger.error(f"Index file not found at {index_path}")
        return []

    file_info_list = []
    for row in soup.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) == 5:  # Assuming a specific table structure
            symbol_tag = cols[2].find('a')
            if symbol_tag and symbol_tag.has_attr('href'):
                file_info = {
                    'Date': cols[1].text.strip(),
                    'Symbol': symbol_tag.text.strip(),
                    'Title': cols[3].find(text=True, recursive=False).strip(),
                    'Originator': cols[4].text.strip(),
                    'file_name': symbol_tag['href']
                }
                file_info_list.append(file_info)
    logger.info(f"Found {len(file_info_list)} files to process in index.")
    return file_info_list

def extract_text_from_pdf(file_path, logger):
    """Extracts all text content from a PDF file."""
    try:
        with fitz.open(file_path) as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {e}")
        return ""

def parse_cleaned_json(file_path, logger):
    """Reads and parses the JSON output from the cleaning API."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_string = f.read().strip()
            # Remove potential markdown code fences
            if json_string.startswith("```json"):
                json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
            
            if json_string:
                return json.loads(json_string)
            return {}
    except FileNotFoundError:
        logger.warning(f"Cleaner output file not found: {file_path}")
    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse JSON from {file_path}: {e}")
    return {}

# --- Core Processing Function ---

def process_files(data_folder, files_to_process, logger):
    """Processes each file: extracts, cleans, and structures data."""
    database = []
    temp_dir = 'temp_for_cleaning'
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_input_path = os.path.join(temp_dir, 'temp_input.txt')
    temp_output_path = os.path.join(temp_dir, 'temp_output.txt')

    for file_info in files_to_process:
        pdf_path = os.path.join(data_folder, file_info['file_name'])
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF file not found: {pdf_path}")
            continue

        logger.info(f"Processing {file_info['file_name']}...")
        
        # 1. Extract raw text from PDF
        raw_content = extract_text_from_pdf(pdf_path, logger)
        if not raw_content:
            continue

        # 2. Write raw text to a temporary file for the API
        with open(temp_input_path, 'w', encoding='utf-8') as f:
            f.write(raw_content)
        
        # 3. Call external API to clean the text
        llm_api_clean(temp_input_path, temp_output_path)
        
        # 4. Read and parse the structured data from the API's output
        # structured_data = parse_cleaned_json(temp_output_path, logger)
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            structured_data = f.read()
        
        # 5. Combine metadata with cleaned data
        entry = {
            **file_info,
            'content': structured_data,
        }
        database.append(entry)

    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    logger.info("Temporary cleaning directory removed.")
            
    return database

# --- Data Saving Functions ---

def save_to_json(data, output_path, logger):
    """Saves the processed data to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Data successfully saved to {output_path}")

def save_to_excel(data, output_path, logger):
    """Saves the processed data to an XLSX file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    logger.info(f"Data successfully saved to {output_path}")

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Setup
    args = parse_arguments()
    logger = setup_logger(args.logging, args.title)
    
    # 2. Define Paths
    data_folder = os.path.join('data', args.title, args.subtitle)
    index_path = os.path.join(data_folder, 'Index.htm')
    output_dir = f'output/{args.title}/{args.subtitle}'
    output_json_path = os.path.join(output_dir, 'data.json')
    output_xlsx_path = os.path.join(output_dir, 'data.xlsx')
    
    # 3. Core Logic
    files_to_process = read_index_file(index_path, logger)
    if files_to_process:
        database = process_files(data_folder, files_to_process, logger)
        
        if database:
            # 4. Save results
            save_to_json(database, output_json_path, logger)
            save_to_excel(database, output_xlsx_path, logger)
        else:
            logger.warning("Database is empty after processing. No files will be saved.")
    else:
        logger.warning("No files to process. Exiting.")
