import pandas as pd
from transformers import AutoTokenizer
import os # For checking file existence

# --- Re-usable Tokenization Function (No special tokens added/skipped) ---
# (Using the version from the previous step)
def tokenize_extract_decode(
    sequence: str,
    tokenizer: AutoTokenizer # Pass the tokenizer object directly for efficiency
):
    """
    Tokenizes a sequence WITHOUT adding special tokens, extracts first/middle/last
    halves, and decodes them WITHOUT skipping any tokens.
    Returns a dictionary with extracted parts, or default empty strings on failure/short sequence.

    Args:
        sequence: The input string sequence to process.
        tokenizer: The initialized Hugging Face tokenizer object.

    Returns:
        A dictionary containing the decoded strings for the first, last, and
        middle halves, and the total token count. Keys are:
        'total_tokens', 'decoded_first_half',
        'decoded_last_half', 'decoded_middle_half'.
        Returns default values (0 tokens, empty strings) if input is empty,
        tokenization fails, or sequence is too short.
    """
    default_result = {
        'total_tokens': 0,
        'decoded_first_half': "",
        'decoded_last_half': "",
        'decoded_middle_half': ""
    }

    if not sequence or not isinstance(sequence, str):
        return default_result

    try:
        # Tokenize without adding special tokens
        token_ids = tokenizer.encode(sequence, add_special_tokens=False)
        num_tokens = len(token_ids)

        # Handle Edge Cases
        if num_tokens < 2:
             return {**default_result, 'total_tokens': num_tokens}

        # Calculate indices
        midpoint = num_tokens // 2
        start_middle = num_tokens // 4
        middle_len = num_tokens // 2
        end_middle = start_middle + middle_len

        # Extract token ID subsets
        first_half_token_ids = token_ids[0:midpoint]
        last_half_token_ids = token_ids[midpoint:]
        middle_half_token_ids = token_ids[start_middle:end_middle]

        # Decode without skipping special tokens
        first_half_string = tokenizer.decode(
            first_half_token_ids, skip_special_tokens=False
        ).strip()
        last_half_string = tokenizer.decode(
            last_half_token_ids, skip_special_tokens=False
        ).strip()
        middle_half_string = tokenizer.decode(
            middle_half_token_ids, skip_special_tokens=False
        ).strip()

        # Return the results
        return {
            'total_tokens': num_tokens,
            'decoded_first_half': first_half_string,
            'decoded_last_half': last_half_string,
            'decoded_middle_half': middle_half_string
        }

    except Exception as e:
        print(f"An error occurred processing sequence '{sequence[:50]}...': {e}")
        return default_result

# --- Main CSV Processing Function (Modified for 3 output files) ---
def process_csv_and_save_splits(
    input_csv_path: str,
    output_first_half_path: str,
    output_middle_half_path: str,
    output_last_half_path: str,
    text_column: str = 'text',
    result_column: str = 'result',
    model_name: str = "bert-base-uncased"
):
    """
    Reads a CSV, processes text using tokenize_extract_decode, and saves
    each extracted half (first, middle, last) along with the original result
    into three separate CSV files.

    Args:
        input_csv_path: Path to the input CSV file.
        output_first_half_path: Path for the CSV with first halves.
        output_middle_half_path: Path for the CSV with middle halves.
        output_last_half_path: Path for the CSV with last halves.
        text_column: Name of the column containing the text to process.
        result_column: Name of the column containing the original result/label.
        model_name: Name of the Hugging Face tokenizer model.
    """
    # --- Input Validation ---
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at '{input_csv_path}'")
        return

    print(f"Loading input CSV: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if text_column not in df.columns:
        print(f"Error: Text column '{text_column}' not found in the input CSV.")
        return
    if result_column not in df.columns:
        print(f"Error: Result column '{result_column}' not found in the input CSV.")
        return

    # --- Initialize Tokenizer (do this once) ---
    print(f"Loading tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer '{model_name}': {e}")
        return

    # --- Process Data ---
    print(f"Processing {len(df)} rows...")
    # *** Key Change: Initialize three lists for output data ***
    first_half_data = []
    middle_half_data = []
    last_half_data = []

    for index, row in df.iterrows():
        original_text = row[text_column]
        # Ensure result is treated as string to avoid potential type issues in CSV
        original_result = str(row[result_column])

        if (index + 1) % 100 == 0:
            print(f"  Processing row {index + 1}/{len(df)}...")

        # Call the extraction function
        extracted_parts = tokenize_extract_decode(original_text, tokenizer)

        # *** Key Change: Populate the three separate lists ***
        # Use a consistent column name like 'extracted_text' for the split part
        first_half_data.append({
            'original_text': original_text,
            'extracted_text': extracted_parts['decoded_first_half'],
            result_column: original_result # Use the original result column name
        })
        middle_half_data.append({
            'original_text': original_text,
            'extracted_text': extracted_parts['decoded_middle_half'],
            result_column: original_result
        })
        last_half_data.append({
            'original_text': original_text,
            'extracted_text': extracted_parts['decoded_last_half'],
            result_column: original_result
        })

    print("Processing complete.")

    # --- Create and Save Output DataFrames ---
    output_columns = ['original_text', 'extracted_text', result_column]

    try:
        print("\nCreating and saving First Half data...")
        first_half_df = pd.DataFrame(first_half_data)
        # Reorder columns for consistency
        first_half_df = first_half_df[output_columns]
        first_half_df.to_csv(output_first_half_path, index=False, encoding='utf-8')
        print(f"Saved First Half data to: {output_first_half_path}")

        print("\nCreating and saving Middle Half data...")
        middle_half_df = pd.DataFrame(middle_half_data)
        middle_half_df = middle_half_df[output_columns]
        middle_half_df.to_csv(output_middle_half_path, index=False, encoding='utf-8')
        print(f"Saved Middle Half data to: {output_middle_half_path}")

        print("\nCreating and saving Last Half data...")
        last_half_df = pd.DataFrame(last_half_data)
        last_half_df = last_half_df[output_columns]
        last_half_df.to_csv(output_last_half_path, index=False, encoding='utf-8')
        print(f"Saved Last Half data to: {output_last_half_path}")

        print("\nAll output files saved successfully.")

    except Exception as e:
        print(f"\nError writing one or more output CSV files: {e}")


# --- Run the script ---
if __name__ == "__main__":
    INPUT_FILE = '/outputs/famous_quotes/output_famous_quotes_12b.csv'  # Your input file path
    TEXT_COL = 'text'             # Column name for text data
    RESULT_COL = 'result'         # Column name for the original result/label
    MODEL = 'EleutherAI/pythia-12b'   # Tokenizer model

    # *** Key Change: Define three output file paths ***
    OUTPUT_DIR = '..'  # Save in the current directory, or specify a path e.g., 'output_splits/'
    # Create output directory if it doesn't exist
    if OUTPUT_DIR != '.' and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    OUTPUT_FIRST = os.path.join(OUTPUT_DIR, 'extracted_first_half_12b.csv')
    OUTPUT_MIDDLE = os.path.join(OUTPUT_DIR, 'extracted_middle_half_12b.csv')
    OUTPUT_LAST = os.path.join(OUTPUT_DIR, 'extracted_last_half_12b.csv')

    process_csv_and_save_splits(
        input_csv_path=INPUT_FILE,
        output_first_half_path=OUTPUT_FIRST,
        output_middle_half_path=OUTPUT_MIDDLE,
        output_last_half_path=OUTPUT_LAST,
        text_column=TEXT_COL,
        result_column=RESULT_COL,
        model_name=MODEL
    )