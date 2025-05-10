import pandas as pd
from tqdm import tqdm
from pathlib import Path
import itertools
import json
from pathlib import Path
from vllm import SamplingParams, LLM
import sys # Added import for sys
from args import ARG
from collections import defaultdict
import time

from llm_prompts import (
    coherence_prompt, repetitive_prompt, outliers_prompt,
    count_outliers_non_words, validate_results, diversity_prompt,
    coverage_prompt, duplicate_concept_prompt, count_duplicate_concept,
    readability_prompt, non_words_detection_prompt, 
    hierarchy_prompt, process_hierarchy_results, 
    semantic_cluster_prompt, complementarity_pair_prompt
)

class OutputBuffer:
    def __init__(self, flush_interval=60):  # Flush every 60 seconds
        self.buffer = defaultdict(dict)  # {file_path: {topic_id: data}}
        self.last_flush = time.time()
        self.flush_interval = flush_interval
        self.cache = {}  # Cache for loaded JSON files

    def add(self, file_path, topic_id, data):
        """Add data to buffer"""
        self.buffer[file_path][str(topic_id)] = data
        
        # Check if we should flush based on time
        try:
            if time.time() - self.last_flush > self.flush_interval:
                self.flush()
        except (TypeError, AttributeError):
            # Handle case where time module might not be available
            self.flush()

    def load_existing_data(self, file_path):
        """Load existing data with caching"""
        if file_path in self.cache:
            return self.cache[file_path]
            
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.cache[file_path] = data
                return data
            except Exception as e:
                print(f"Warning: Could not load existing data from {file_path}: {e}")
        return {}

    def flush(self):
        """Write buffered data to files"""
        if not self.buffer:  # Skip if buffer is empty
            return

        for file_path, topic_data in self.buffer.items():
            if not topic_data:
                continue
                
            # Load existing data
            existing_data = self.load_existing_data(file_path)
            
            # Update with buffered data
            existing_data.update(topic_data)
            
            # Write updated data
            try:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                self.cache[file_path] = existing_data  # Update cache
            except Exception as e:
                print(f"Error saving to {file_path}: {e}")

        # Clear buffer and update flush time
        self.buffer.clear()
        try:
            self.last_flush = time.time()
        except (TypeError, AttributeError):
            pass  # Ignore if time module is not available

    def __del__(self):
        """Ensure all data is flushed when object is destroyed"""
        try:
            self.flush()
        except Exception as e:
            # Silently handle errors during destruction
            pass

# Create a global buffer instance
output_buffer = OutputBuffer()

def get_file_dir(path):
    return list(Path(path).rglob('*_0.csv'))
#[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

def load_topic_words_files(file_path):
    # Ensure 'ID' column exists before setting it as index
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        if 'ID' in df.columns:
            df = df.set_index('ID')
        else:
            # Handle cases where ID might be the first unnamed column from previous saves
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8', index_col=0)
            if df.index.name != 'ID':
                 print(f"Warning: Could not find 'ID' column in {file_path}. Using first column as index.")
                 df.index.name = 'ID' # Assign a name if needed
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        # Return an empty DataFrame or raise error as appropriate
        return pd.DataFrame()
    
def get_dataset_name(data_source):
    """Extract dataset name from data source path"""
    return data_source.split('/')[-1]

def get_topic_words(df):
    if 'topic words' not in df.columns:
        print("Warning: 'topic words' column not found in DataFrame.")
        if 'Topic words' in df.columns:
            df.rename(columns={'Topic words': 'topic words'}, inplace=True)
            print("Renamed 'Topic words' to 'topic words' for consistency.")
            return df['topic words'].dropna().apply(lambda x: ', '.join([f"'{i}'" for i in str(x).split()])).tolist()
        return []
    # Handle potential NaN values gracefully
    return df['topic words'].dropna().apply(lambda x: ', '.join([f"'{i}'" for i in str(x).split()])).tolist()

def count_coverage(results, repeat=5):
    if not results: # Handle empty results
        return []
    if len(results) % repeat != 0:
        print(f"Warning: coverage result numbers ({len(results)}) not divisible by repeat ({repeat}).")
        # Decide how to handle this: return empty, raise error, or process partial data?
        # For now, let's process as much as possible
        pass # Or adjust logic as needed

    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    word_num = []
    for numbers in chunk(results, repeat):
        try:
            if isinstance(numbers[0], str):
                # Added error handling for parsing
                parsed_numbers = []
                for ids in numbers:
                    try:
                        # Handle potential format variations like empty strings or missing brackets
                        content = ids.split(']')[0]
                        if '[' in content:
                            content = content.split('[')[1]
                        if content.strip(): # Check if content is not empty after stripping
                            parsed_numbers.append(len(content.split(',')))
                        else:
                            parsed_numbers.append(0) # Treat empty content as 0 words
                    except IndexError:
                         parsed_numbers.append(0) # Handle cases where split fails
                numbers = parsed_numbers

            if not numbers: # Handle empty chunk after parsing
                 word_num.append(0)
                 continue

            if sum(numbers) == 0:
                word_num.append(0)
            else:
                non_zero = [n for n in numbers if n > 0]
                if not non_zero: # Handle case where all numbers are zero
                    word_num.append(0)
                else:
                    word_num.append(sum(non_zero) / len(non_zero))
        except Exception as e:
            print(f"Error processing chunk {numbers}: {e}")
            word_num.append(0) # Append a default value or handle error appropriately
    return word_num

# VLLM Prompting Helpers
def run_vllm(llm, prompts, max_tokens, temperature=0.7, top_p=0.95, n=1):
    if not prompts: # Handle empty prompt list
        return []
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n
    )
    outputs = llm.generate(prompts, sampling_params)
    # Handle cases where outputs might be empty or malformed
    return [o.outputs[0].text.strip() for o in outputs if o.outputs]

def batch_prompting(llm, prompts, max_tokens):
    return run_vllm(llm, prompts, max_tokens)

def batch_prompting_with_validate(llm, prompts, max_tokens, rating_method, batch_size=4096):
    results = []
    if not prompts:  # Handle empty prompt list
        return results

    def chunks(lst, batch_size):
        """Yield successive batch_size chunks from lst."""
        if not lst:
            return
        try:
            for i in range(0, len(lst), batch_size):
                yield lst[i:i + batch_size]  # Fixed slice syntax
        except Exception as e:
            print(f"Error in chunks function: {str(e)}")
            print(f"List length: {len(lst)}, batch_size: {batch_size}, current i: {i}")
            raise

    for batch in tqdm(list(chunks(prompts, batch_size)), desc="Batch Prompting w/ Validate"):
        if not batch: 
            continue

        outputs = run_vllm(llm, batch, max_tokens)
        if not outputs:
            print("Warning: Received empty outputs from run_vllm for a batch.")
            results.extend([None] * len(batch))
            continue

        # Initial validation
        if rating_method == "count":
            try:
                # Added error handling for parsing numbers
                numbers = []
                for ids in outputs:
                    try:
                        content = ids.split(']')[0]
                        if '[' in content:
                            content = content.split('[')[1]
                        if content.strip():
                            numbers.append(len(content.split(',')))
                        else:
                            numbers.append(0)
                    except IndexError:
                        numbers.append(-1) # Indicate parsing error
                # Check if numbers list is empty before using max/min
                retry = numbers and (max(numbers) > 15 or min(numbers) < 0)
            except Exception as e:
                 print(f"Error parsing numbers for validation: {e}. Outputs: {outputs}")
                 retry = True # Retry if parsing fails
        else:
            retry, validated_outputs = validate_results(outputs, "coverage", rating_method)
            outputs = validated_outputs # Use validated outputs

        tries = 0
        while retry and tries < 5:
            tries += 1
            print(f"Invalid generation (try {tries}/5), retrying batch...")
            outputs = run_vllm(llm, batch, max_tokens)
            if not outputs:
                print("Warning: Received empty outputs during retry.")
                # Decide how to handle retry failure
                break # Exit retry loop if LLM consistently fails

            if rating_method == "count":
                try:
                    numbers = []
                    for ids in outputs:
                        try:
                            content = ids.split(']')[0]
                            if '[' in content:
                                content = content.split('[')[1]
                            if content.strip():
                                numbers.append(len(content.split(',')))
                            else:
                                numbers.append(0)
                        except IndexError:
                            numbers.append(-1)
                    retry = numbers and (max(numbers) > 15 or min(numbers) < 0)
                except Exception as e:
                    print(f"Error parsing numbers during retry: {e}. Outputs: {outputs}")
                    retry = True # Continue retrying if parsing fails
            else:
                retry, validated_outputs = validate_results(outputs, "coverage", rating_method)
                outputs = validated_outputs

        if retry: # If still failing after retries
             print(f"Warning: Batch failed validation after 5 retries. Using last output or placeholders.")
             # Add placeholders or last potentially invalid output
             results.extend(outputs if outputs else [None] * len(batch))
        else:
             results.extend(outputs)

    # Ensure results length matches prompts length if placeholders were used
    if len(results) != len(prompts):
         print(f"Warning: Mismatch in results length ({len(results)}) and prompts length ({len(prompts)}). Adjusting...")
         # This might indicate an issue needing investigation.
         # For now, pad with None or truncate if necessary.
         results.extend([None] * (len(prompts) - len(results)))
         results = results[:len(prompts)]

    return results

def batch_prompting_with_repeat(llm, prompts, max_tokens, rating_method, metric, repeat=5):
    """
    Run batch prompting multiple times and return both processed results and raw outputs
    """
    all_results = []
    all_raw_outputs = [[] for _ in range(len(prompts))]
    
    for run in range(repeat):
        results = batch_prompting(llm, prompts, max_tokens)
        retry, validated_results = validate_results(results, metric, rating_method)
        
        tries = 0
        while retry and tries < 5:
            tries += 1
            print(f"Invalid generation (try {tries}/5), retrying...")
            results = batch_prompting(llm, prompts, max_tokens)
            retry, validated_results = validate_results(results, metric, rating_method)
        
        if retry:
            print(f"Warning: Validation failed after 5 retries for {metric}")
            continue
        
        # Store both validated results and raw outputs
        all_results.append(validated_results)
        for i, raw_output in enumerate(results):
            if i < len(all_raw_outputs):
                all_raw_outputs[i].append(raw_output)
    
    # Process results
    if not all_results:
        return [None] * len(prompts), [[] for _ in range(len(prompts))]
    
    averaged_results = []
    for i in range(len(prompts)):
        valid_scores = [results[i] for results in all_results if results and i < len(results)]
        averaged_results.append(sum(valid_scores) / len(valid_scores) if valid_scores else None)
    
    return averaged_results, all_raw_outputs

def save_raw_outputs_by_id(save_path_dir, file_stem, topic_id, raw_outputs):
    """
    Optimized version using buffer for batch operations
    """
    raw_output_dir = save_path_dir / "raw_outputs"
    output_file = raw_output_dir / f"{file_stem}_raw.json"
    
    # Add to buffer instead of immediate write
    output_buffer.add(str(output_file), topic_id, raw_outputs)

def main(args):
    try:
        model_config = {
            "mistral": {
                "path": "TouchNight/Ministral-8B-Instruct-2410-HF",
                "args":{
                    "max_model_len": 8192,
                    # "load_format": "mistral", # Keep commented unless needed
                }
            },
            "mistral-large": {
                "path": "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit",
                "args":{
                    "max_model_len": 8192,
                    "load_format": "bitsandbytes",
                }
            },
            "llama": {
                "path": "unsloth/Meta-Llama-3.1-8B-Instruct"
            },
            "llama-large": {
                "path": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
                "args":{
                    "max_model_len": 8192,
                    "load_format": "bitsandbytes",
                }
            },
            "qwen": {
                "path": "Qwen/Qwen2.5-7B-Instruct"
            },
            "qwen-large": {
                "path": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
                "args":{
                    "max_model_len": 8192,
                    "load_format": "bitsandbytes",
                    # "tensor_parallel_size": 2
                }
            },
            "gemma":{
                "path": "google/gemma-3-4b-it",
            },
            "gemma-large":{
                "path": "google/gemma-3-27b-it",
                "args":{
                    "max_model_len": 8192
                }
            }
        }

        model_source = args.model
        if model_source not in model_config:
            raise ValueError(f"Model '{model_source}' not found in configuration. Available models: {list(model_config.keys())}")

        model_path = model_config[model_source]["path"]
        model_args = model_config[model_source].get("args", {})

        print(f"\nAttempting to load model:")
        print(f"- Model source: {model_source}")
        print(f"- Model path: {model_path}")
        print(f"- Model arguments: {model_args}")

        try:
            llm = LLM(model=model_path, **model_args)
        except ValueError as ve:
            raise ValueError(f"Invalid argument when loading model: {str(ve)}\n"
                           f"Model path: {model_path}\n"
                           f"Model arguments: {model_args}")
        except RuntimeError as re:
            raise RuntimeError(f"Runtime error loading model: {str(re)}\n"
                             f"This might be due to insufficient GPU memory or invalid model configuration.")
        except Exception as e:
            raise Exception(f"Unexpected error loading VLLM model:\n"
                          f"Error type: {type(e).__name__}\n"
                          f"Error message: {str(e)}\n"
                          f"Model path: {model_path}\n"
                          f"Model arguments: {model_args}")

        print("Model loaded successfully!")

        rating_method = args.rating
        dataset_name = get_dataset_name(args.data_source)
        result_path = Path('data/llm_judgement_output') / dataset_name

        # --- Determine Max Tokens based on Rating Method ---
        if rating_method == "choose":
            max_new_tokens = 5
        elif rating_method == "bool":
            max_new_tokens = 3
        elif rating_method == "number":
            max_new_tokens = 1
        else: # Default or for count/other methods if needed
            max_new_tokens = 10 # Adjust default as necessary

        repeat = 5

        # --- Coverage Processing ---
        if args.coverage:
            coverage_repeat = 5
            print(f"--- Starting Coverage Processing for {dataset_name} (repeat={coverage_repeat}) ---")
            coverage_result_dir = result_path.parent / "coverage_results" / model_source
            coverage_result_dir.mkdir(exist_ok=True, parents=True)

            for file_path in Path(args.data_source).glob('sample_100_data*'):
                try:
                    print(f"Processing coverage for file: {file_path}")
                    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
                    if 'text' not in df.columns or 'topic words' not in df.columns:
                        print(f"Skipping {file_path}: Missing required columns 'text' or 'topic words'.")
                        continue

                    topic_words_list = get_topic_words(df)
                    documents = df['text'].tolist()

                    if len(topic_words_list) != len(documents):
                        print(f"Skipping {file_path}: Mismatch between topic words ({len(topic_words_list)}) and documents ({len(documents)}).")
                        continue
                    if not topic_words_list:
                        print(f"Skipping {file_path}: No valid topic words found.")
                        continue

                    # Add model information to DataFrame
                    df['llm_model'] = model_source  # Track which LLM made the evaluation

                    coverage_prompts = [coverage_prompt(tw, doc, model_source, rating_method) for tw, doc in zip(topic_words_list, documents)]
                    valid_prompts = [(p, i) for i, p in enumerate(coverage_prompts) if p is not None and len(p) == 2]
                    if not valid_prompts:
                        print(f"Skipping {file_path}: No valid coverage prompts generated.")
                        continue

                    original_indices = [i for p, i in valid_prompts]
                    prompts_tuples = [p for p, i in valid_prompts]

                    coverage_prompts_low = [p[0] for p in prompts_tuples for _ in range(coverage_repeat)]
                    coverage_prompts_high = [p[1] for p in prompts_tuples for _ in range(coverage_repeat)]

                    print(f"Running batch prompting for low coverage using {model_source}...")
                    coverage_result_low = batch_prompting_with_validate(llm, coverage_prompts_low, max_tokens=30, rating_method="count")
                    print(f"Running batch prompting for high coverage using {model_source}...")
                    coverage_result_high = batch_prompting_with_validate(llm, coverage_prompts_high, max_tokens=30, rating_method="count")

                    # Align results back to the original DataFrame indices
                    under_coverage_examples = [coverage_result_low[i*coverage_repeat:(i+1)*coverage_repeat] for i in range(len(prompts_tuples))]
                    over_coverage_examples = [coverage_result_high[i*coverage_repeat:(i+1)*coverage_repeat] for i in range(len(prompts_tuples))]
                    under_coverage_scores = count_coverage(coverage_result_low, repeat=coverage_repeat)
                    over_coverage_scores = count_coverage(coverage_result_high, repeat=coverage_repeat)

                    # Create temporary DataFrame for results with model information
                    results_df = pd.DataFrame({
                        'llm_model_new': model_source,  # Changed column name to avoid conflict
                        'under_coverage_example': under_coverage_examples,
                        'over_coverage_example': over_coverage_examples,
                        'under_coverage': under_coverage_scores,
                        'over_coverage': over_coverage_scores
                    }, index=df.index[original_indices])

                    # Merge results back into the original DataFrame with suffix handling
                    df = df.join(results_df, how='left', lsuffix='_old', rsuffix='_new')
                    
                    # Clean up column names if needed
                    if 'llm_model_old' in df.columns:
                        df.drop('llm_model_old', axis=1, inplace=True)
                    if 'llm_model_new' in df.columns:
                        df.rename(columns={'llm_model_new': 'llm_model'}, inplace=True)

                    # Save results with model name in filename
                    coverage_output_file = coverage_result_dir / f"{file_path.stem}_coverage_results_{model_source}.csv"
                    print(f"Saving combined coverage results to: {coverage_output_file}")
                    df.to_csv(coverage_output_file, sep='\t', index=True)

                    # Save raw outputs with model information
                    for idx, (under_examples, over_examples, under_score, over_score) in enumerate(zip(
                        under_coverage_examples, over_coverage_examples, 
                        under_coverage_scores, over_coverage_scores)):
                        
                        id_raw_outputs = {
                            "coverage": {
                                "llm_model": model_source,
                                "processed": {
                                    "under_coverage": under_score,
                                    "over_coverage": over_score
                                },
                                "raw": {
                                    "under_coverage": under_examples,
                                    "over_coverage": over_examples
                                }
                            }
                        }
                        save_raw_outputs_by_id(coverage_result_dir, Path(file_path).stem, idx, id_raw_outputs)
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
            return # Exit after coverage processing

        # --- Diversity Processing ---
        elif args.diversity:
            print(f"--- Starting Diversity Processing for {dataset_name} ---")
            diversity_result_dir = result_path / model_source / rating_method / "diversity"
            diversity_result_dir.mkdir(parents=True, exist_ok=True)

            file_path_list = get_file_dir(args.data_source)
            if not file_path_list:
                print(f"No '*_0.csv' files found in {args.data_source}")
                return

            for file_path in tqdm(file_path_list, desc="Processing Diversity"):
                print(f"Processing diversity for: {file_path}")
                df = load_topic_words_files(file_path)
                if df.empty:
                    continue  # Skip if loading failed

                topic_words_list = get_topic_words(df)
                if len(topic_words_list) < 2:
                    print(f"Skipping {file_path}: Need at least 2 topics for diversity calculation.")
                    continue

                # Generate topic word pairs and prompts
                pairs = list(itertools.combinations(topic_words_list, 2))
                prompts = [diversity_prompt(pair, model_source, rating_method) for pair in pairs]
                valid_prompts_data = [(p, pair) for p, pair in zip(prompts, pairs) if p is not None]
                if not valid_prompts_data:
                    print(f"Skipping {file_path}: No valid diversity prompts generated.")
                    continue

                valid_prompts = [p for p, pair in valid_prompts_data]
                valid_pairs = [pair for p, pair in valid_prompts_data]

                # Run diversity analysis
                results = batch_prompting(llm, valid_prompts, max_tokens=max_new_tokens)
                retry, validated_results = validate_results(results, "diversity", rating_method)
                tries = 0
                while retry and tries < 5:
                    tries += 1
                    print(f"Invalid diversity generation (try {tries}/5), retrying...")
                    results = batch_prompting(llm, valid_prompts, max_tokens=max_new_tokens)
                    retry, validated_results = validate_results(results, "diversity", rating_method)

                if retry:
                    print(f"Warning: Diversity validation failed for {file_path} after 5 retries.")
                    continue

                results = validated_results  # Use validated results

                # Add hierarchy analysis for three variants
                hierarchy_all_outputs = []
                hierarchy_all_results = []
                for variant in range(3):  # Taxonomic, Aspectual, Containment
                    hierarchy_prompts = [
                        hierarchy_prompt(pair, model_source, rating_method, variant=variant)
                        for pair in valid_pairs
                    ]
                    variant_outputs = batch_prompting(llm, hierarchy_prompts, max_tokens=50)
                    variant_results = [process_hierarchy_results(output) for output in variant_outputs]
                    hierarchy_all_outputs.append(variant_outputs)
                    hierarchy_all_results.append(variant_results)

                # Store both processed and raw results
                stem = Path(file_path).stem
                for i, (pair, div_score, h_results, h_outputs) in enumerate(zip(
                    valid_pairs, 
                    results, 
                    zip(*hierarchy_all_results),  # Processed results for each variant
                    zip(*hierarchy_all_outputs)   # Raw outputs for each variant
                )):
                    id_raw_outputs = {
                        "diversity": {
                            "processed": {
                                "score": div_score,
                                "pair": list(pair),
                                "hierarchy": {
                                    "taxonomic": {
                                        "is_hierarchical": h_results[0][0],
                                        "direction": h_results[0][1]
                                    },
                                    "aspectual": {
                                        "is_hierarchical": h_results[1][0],
                                        "direction": h_results[1][1]
                                    },
                                    "containment": {
                                        "is_hierarchical": h_results[2][0],
                                        "direction": h_results[2][1]
                                    }
                                }
                            },
                            "raw": {
                                "diversity_output": results[i],
                                "hierarchy_outputs": {
                                    "taxonomic": h_outputs[0],
                                    "aspectual": h_outputs[1],
                                    "containment": h_outputs[2]
                                }
                            }
                        }
                    }
                    save_raw_outputs_by_id(diversity_result_dir, stem, i, id_raw_outputs)

                try:
                    with open(diversity_result_dir / (stem + '.json'), 'w') as f:
                        summary_results = [{
                            "diversity_score": r,
                            "pair": list(p),
                            "is_hierarchical": h[0],
                            "direction": h[1]
                        } for p, r, h in zip(valid_pairs, results, zip(*hierarchy_all_results))]
                        json.dump(summary_results, f, indent=2)
                except Exception as e:
                    print(f"Error saving diversity results for {file_path}: {e}")

                # Save summary results
                try:
                    with open(diversity_result_dir / (stem + '.json'), 'w') as f:
                        json.dump(results, f)
                    with open(diversity_result_dir / (stem + '_pairs.json'), 'w') as f:
                        # Save valid pairs corresponding to the results
                        json.dump([[list(p), r] for p, r in zip(valid_pairs, results)], f, indent=2)
                    if results:  # Avoid division by zero
                        avg_score = sum(filter(None, results)) / len([r for r in results if r is not None])
                        with open(diversity_result_dir / (stem + '_average.txt'), 'w') as f:
                            f.write(str(avg_score))
                    else:
                        with open(diversity_result_dir / (stem + '_average.txt'), 'w') as f:
                            f.write("N/A (No results)")

                except Exception as e:
                    print(f"Error saving diversity results for {file_path}: {e}")

        # --- Coherence / Repetitive Processing ---
        elif args.coherence or args.repetitive or args.readability:
            print(f"--- Starting Coherence/Repetitive/Readability Processing for {dataset_name} ---")
            file_path_list = get_file_dir(args.data_source)
            if not file_path_list:
                print(f"No '*_0.csv' files found in {args.data_source}")
                return

            # Define the desired final order of result columns
            # We'll add the original columns at the beginning later
            result_column_order = [
                "Coherence",
                "Repetitive",
                "Readability",
                "Number of Outliers",
                "Outliers",
                "Number of Same Concept Pairs",
                "Same Concept Pairs",
                "Number of Non-words",
                "Non-words",
            ]

            for file_path in tqdm(file_path_list, desc="Processing Coherence/Repetitive/Readability"):
                print(f"Processing file: {file_path}")
                # 1. Load base data first
                try:
                    df_base = load_topic_words_files(file_path)
                    if df_base.empty:
                        print(f"  Skipping {file_path}: Loaded empty DataFrame.")
                        continue
                    # Store original columns (excluding the index if it's named 'ID')
                    original_columns = [col for col in df_base.columns if col != df_base.index.name]
                except Exception as e:
                    print(f"  Error loading base file {file_path}: {e}. Skipping.")
                    continue

                topic_words_list = get_topic_words(df_base)
                if not topic_words_list:
                    print(f"  Skipping {file_path}: No valid topic words found.")
                    continue

                # 2. Define output path with dataset name
                save_path_dir = result_path / model_source / rating_method
                save_path_dir.mkdir(parents=True, exist_ok=True)
                output_file_path = save_path_dir / Path(file_path).name

                # 3. Initialize result holders
                current_run_results = {} # Store results calculated in *this* run

                # Initialize raw outputs collector for this ID
                current_raw_outputs = {
                    "coherence": {},
                    "repetitive": {},
                    "readability": {},
                    "outliers": {},
                    "duplicate_concept": {},
                    "non_words": {}
                }

                # --- Coherence & Outliers ---
                if args.coherence:
                    print(f"  Processing coherence & outliers...")
                    coherence_prompts = [coherence_prompt(tw, model_source, rating_method) for tw in topic_words_list]
                    outliers_prompts = [p for tw in topic_words_list for p in outliers_prompt(model_source, tw, repeat=repeat)]

                    # Filter None prompts and track indices (simplified example, use your existing robust filtering)
                    valid_coherence_prompts = [(p, i) for i, p in enumerate(coherence_prompts) if p is not None]
                    valid_outliers_prompts = [(p, i // repeat) for i, p in outliers_prompts if p is not None]

                    coherence_results_list = [None] * len(df_base)
                    outliers_num_list = [None] * len(df_base)
                    outliers_word_list = [None] * len(df_base)

                    if valid_coherence_prompts:
                        c_prompts = [p for p, i in valid_coherence_prompts]
                        c_indices = [i for p, i in valid_coherence_prompts]
                        
                        c_results, c_results_raw = batch_prompting_with_repeat(
                            llm, c_prompts, max_new_tokens, 
                            rating_method, "coherence", repeat=repeat
                        )
                        
                        # Store both processed and raw results
                        current_raw_outputs["coherence"] = {
                            "processed": {idx: score for idx, score in zip(c_indices, c_results)},
                            "raw": {idx: raw for idx, raw in zip(c_indices, c_results_raw)}
                        }

                        coherence_results_list = [None] * len(df_base)
                        for res, idx in zip(c_results, c_indices):
                            coherence_results_list[idx] = res
                        
                        current_run_results["Coherence"] = coherence_results_list

                    if valid_outliers_prompts:
                        o_prompts = [p for p, i in valid_outliers_prompts]
                        o_outputs_raw = batch_prompting(llm, o_prompts, max_tokens=50)
                        o_num, o_word = count_outliers_non_words(o_outputs_raw, repeat=repeat)
                        
                        current_raw_outputs["outliers"] = {
                            "processed": {
                                idx: {"num": num, "words": words}
                                for idx, (num, words) in enumerate(zip(o_num, o_word))
                            },
                            "raw": {
                                idx: raw_outputs
                                for idx, raw_outputs in enumerate(
                                    [o_outputs_raw[i:i+repeat] for i in range(0, len(o_outputs_raw), repeat)]
                                )
                            }
                        }

                        if len(o_num) == len(df_base):
                             outliers_num_list = o_num
                             outliers_word_list = o_word
                        else:
                             print(f"  Warning: Outlier result length mismatch ({len(o_num)}). Expected {len(df_base)}.")

                        current_run_results["Number of Outliers"] = outliers_num_list
                        current_run_results["Outliers"] = outliers_word_list


                # --- Repetitive & Same Concept ---
                if args.repetitive:
                    print(f"  Processing repetitive & same concept...")
                    repetitive_prompts = [repetitive_prompt(tw, model_source, rating_method) for tw in topic_words_list]
                    duplicate_concept_prompts = [duplicate_concept_prompt(model_source, tw) for tw in topic_words_list]

                    # Filter None prompts and track indices (simplified example)
                    valid_repetitive_prompts = [(p, i) for i, p in enumerate(repetitive_prompts) if p is not None]
                    valid_duplicate_concept_prompts = [(p, i) for i, p in enumerate(duplicate_concept_prompts) if p is not None]

                    repetitive_results_list = [None] * len(df_base)
                    duplicate_concept_num_list = [None] * len(df_base)
                    duplicate_concept_words_list = [None] * len(df_base)

                    if valid_repetitive_prompts:
                        r_prompts = [p for p, i in valid_repetitive_prompts]
                        r_indices = [i for p, i in valid_repetitive_prompts]
                        
                        r_results, r_results_raw = batch_prompting_with_repeat(
                            llm, r_prompts, max_new_tokens, 
                            rating_method, "repetitive", repeat=repeat
                        )
                        
                        current_raw_outputs["repetitive"] = {
                            "processed": {idx: score for idx, score in zip(r_indices, r_results)},
                            "raw": {idx: raw for idx, raw in zip(r_indices, r_results_raw)}
                        }

                        repetitive_results_list = [None] * len(df_base)
                        for res, idx in zip(r_results, r_indices):
                            repetitive_results_list[idx] = res
                            
                        current_run_results["Repetitive"] = repetitive_results_list

                    if valid_duplicate_concept_prompts:
                        sc_prompts = [p for p, i in valid_duplicate_concept_prompts]
                        sc_indices = [i for p, i in valid_duplicate_concept_prompts]
                        sc_output_raw = batch_prompting(llm, sc_prompts, max_tokens=50)
                        full_sc_output = [None] * len(df_base) # Map results back correctly
                        
                        # Store raw outputs before processing
                        current_raw_outputs["duplicate_concept"] = {
                            "processed": {},
                            "raw": {}
                        }
                        
                        for res, idx in zip(sc_output_raw, sc_indices):
                            full_sc_output[idx] = res
                            # Store raw output by index
                            current_raw_outputs["duplicate_concept"]["raw"][idx] = res
                        
                        sc_num, sc_words = count_duplicate_concept(full_sc_output)
                        if len(sc_num) == len(df_base):
                            duplicate_concept_num_list = sc_num
                            duplicate_concept_words_list = sc_words
                            
                            # Store processed results
                            for idx in range(len(df_base)):
                                current_raw_outputs["duplicate_concept"]["processed"][idx] = {
                                    "num_pairs": sc_num[idx],
                                    "word_pairs": sc_words[idx]
                                }
                        else:
                            print(f"  Warning: Same Concept result length mismatch ({len(sc_num)}). Expected {len(df_base)}.")

                        current_run_results["Number of Same Concept Pairs"] = duplicate_concept_num_list
                        current_run_results["Same Concept Pairs"] = duplicate_concept_words_list
                
                # --- Readability & Non-words Detection ---
                if args.readability:
                    print(f"  Processing readability & non-words detection...")
                    readability_prompts = [readability_prompt(tw, model_source, rating_method) for tw in topic_words_list]
                    non_words_prompts = [p for tw in topic_words_list for p in non_words_detection_prompt(model_source, tw, repeat=repeat)]

                    # Filter None prompts and track indices (simplified example, use your existing robust filtering)
                    valid_readability_prompts = [(p, i) for i, p in enumerate(readability_prompts) if p is not None]
                    valid_non_words_prompts = [(p, i // repeat) for i, p in non_words_prompts if p is not None]

                    readability_results_list = [None] * len(df_base)
                    non_words_num_list = [None] * len(df_base)
                    non_words_word_list = [None] * len(df_base)

                    if valid_readability_prompts:
                        re_prompts = [p for p, i in valid_readability_prompts]
                        re_indices = [i for p, i in valid_readability_prompts]
                        
                        re_results, re_results_raw = batch_prompting_with_repeat(
                            llm, re_prompts, max_new_tokens, 
                            rating_method, "readability", repeat=repeat
                        )
                        
                        current_raw_outputs["readability"] = {
                            "processed": {idx: score for idx, score in zip(re_indices, re_results)},
                            "raw": {idx: raw for idx, raw in zip(re_indices, re_results_raw)}
                        }
                        
                        readability_results_list = [None] * len(df_base)
                        for res, idx in zip(re_results, re_indices):
                            readability_results_list[idx] = res
                        
                        current_run_results["Readability"] = readability_results_list

                    if valid_non_words_prompts:
                        n_prompts = [p for p, i in valid_non_words_prompts]
                        n_outputs_raw = batch_prompting(llm, n_prompts, max_tokens=50)
                        n_num, n_word = count_outliers_non_words(n_outputs_raw, repeat=repeat)
                        
                        current_raw_outputs["non_words"] = {
                            "processed": {
                                idx: {"num": num, "words": words}
                                for idx, (num, words) in enumerate(zip(n_num, n_word))
                            },
                            "raw": {
                                idx: raw_outputs
                                for idx, raw_outputs in enumerate(
                                    [n_outputs_raw[i:i+repeat] for i in range(0, len(n_outputs_raw), repeat)]
                                )
                            }
                        }

                        if len(n_num) == len(df_base):
                            non_words_num_list = n_num
                            non_words_word_list = n_word
                        else:
                            print(f"  Warning: Non-words result length mismatch ({len(o_num)}). Expected {len(df_base)}.")

                        current_run_results["Number of Non-words"] = non_words_num_list
                        current_run_results["Non-words"] = non_words_word_list

                # After processing all tasks for this ID, save the raw outputs
                file_stem = Path(file_path).stem
                for idx in df_base.index:
                    id_raw_outputs = {
                        task: {
                            "processed": data["processed"].get(idx),
                            "raw": data["raw"].get(idx)
                        }
                        for task, data in current_raw_outputs.items()
                        if data.get("processed", {}).get(idx) is not None
                    }
                    if any(id_raw_outputs.values()):  # Only save if there are any results
                        save_raw_outputs_by_id(save_path_dir, file_stem, idx, id_raw_outputs)

                # --- Combine Results and Save ---
                if not current_run_results:
                    print(f"  No results generated for {file_path} in this run.")
                    continue # Skip saving if nothing was calculated

                # 4. Load existing results if file exists
                final_df = df_base.copy() # Start with the original data for this file
                if output_file_path.exists():
                    print(f"  Merging with existing file: {output_file_path}")
                    try:
                        existing_df = pd.read_csv(output_file_path, sep='\t', index_col='ID')
                        # Add columns from existing file that were *not* calculated in this run
                        for col in result_column_order:
                            if col in existing_df.columns and col not in current_run_results:
                                # Use join to align by index, handling potential missing indices
                                final_df = final_df.join(existing_df[[col]])
                    except Exception as e:
                        print(f"  Warning: Could not load or merge existing file {output_file_path}: {e}. Overwriting with current results.")
                        # Reset final_df to base if loading failed badly
                        final_df = df_base.copy()


                # 5. Add/Update with results calculated in *this* run
                for col_name, data in current_run_results.items():
                    if len(data) == len(final_df.index):
                        final_df[col_name] = data
                    else:
                        print(f"  Warning: Length mismatch for new column '{col_name}' ({len(data)}) and DataFrame index ({len(final_df.index)}). Skipping assignment.")


                # 6. Define final column order and select existing columns
                # Start with original columns, then add result columns in desired order
                full_desired_order = original_columns + result_column_order
                # Filter the order list to only include columns actually present in the final DataFrame
                columns_to_save = [col for col in full_desired_order if col in final_df.columns]

                # 7. Save the combined and ordered results
                print(f"  Saving results to: {output_file_path}")
                try:
                    # Select columns in the desired order before saving
                    final_df[columns_to_save].to_csv(output_file_path, sep="\t", index=True) # Keep index
                except Exception as e:
                    print(f"  Error saving results for {file_path}: {e}")

        # Add after other task blocks (coherence/repetitive/readability)
        elif args.clustering_complementarity:
            print(f"--- Starting Clustering Complementarity Processing for {dataset_name} ---")
            clustering_result_dir = result_path / model_source / "clustering_complementarity"
            clustering_result_dir.mkdir(parents=True, exist_ok=True)

            file_path_list = get_file_dir(args.data_source)
            if not file_path_list:
                print(f"No '*_0.csv' files found in {args.data_source}")
                return

            for file_path in tqdm(file_path_list, desc="Processing Clustering Complementarity"):
                print(f"Processing file: {file_path}")
                try:
                    df_base = load_topic_words_files(file_path)
                    if df_base.empty:
                        print(f"  Skipping {file_path}: Loaded empty DataFrame.")
                        continue

                    topic_words_list = get_topic_words(df_base)
                    if not topic_words_list:
                        print(f"  Skipping {file_path}: No valid topic words found.")
                        continue

                    # Initialize results storage
                    current_raw_outputs = {
                        "semantic_clustering": {},
                        "complementarity": {}
                    }

                    # Process semantic clustering for all topics at once
                    semantic_prompt = semantic_cluster_prompt(topic_words_list, model_source)
                    comp_prompt = complementarity_pair_prompt(topic_words_list, model_source)
                    if semantic_prompt and comp_prompt:
                        # Run semantic clustering analysis
                        cs_results_raw = run_vllm(
                            llm, [semantic_prompt, comp_prompt], 4096
                        )

                        if cs_results_raw:
                            # Store raw outputs
                            current_raw_outputs["semantic_clustering"] = {
                                "raw": {"overall_output": cs_results_raw[0]}
                            }
                            current_raw_outputs["complementarity"] = {
                                "raw": {"overall_output": cs_results_raw[1]}
                            }
                            json.dump(current_raw_outputs, open(clustering_result_dir / (Path(file_path).stem + '_raw.json'), 'w'), indent=2)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

        else:
             print("No task selected. Use --coverage, --diversity, --coherence, or --repetitive.")
    
    except Exception as e:
        print("\nCritical error in main execution:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        if hasattr(e, '__cause__') and e.__cause__ is not None:
            print("\nCaused by:")
            print(f"Error type: {type(e.__cause__).__name__}")
            print(f"Error message: {str(e.__cause__)}")
        
        print("\nDebug information:")
        print(f"Model source: {args.model}")
        if 'model_path' in locals():
            print(f"Model path: {model_path}")
        if 'model_args' in locals():
            print(f"Model arguments: {model_args}")
            
        raise  # Re-raise the exception for the main error handler
    finally:
        # Ensure all buffered data is written
        output_buffer.flush()


# --- Argument Parsing Setup ---
if __name__ == "__main__":
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            raise ValueError(
                "Incorrect number of arguments.\n"
                "Usage: python evaluation.py <config.json>\n"
                "Config JSON should contain keys: model, data_source, rating, "
                "[diversity, coverage, coherence, repetitive]"
            )

        config_path = sys.argv[1]
        
        # Load and validate config file
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                arg_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in config file: {e.msg}",
                e.doc,
                e.pos
            )
        except Exception as e:
            raise RuntimeError(f"Error reading config file {config_path}: {str(e)}")

        # Validate required configuration keys
        required_keys = ["model", "data_source", "rating"]
        missing_keys = [key for key in required_keys if key not in arg_dict]
        if missing_keys:
            raise KeyError(
                f"Missing required keys in config file: {', '.join(missing_keys)}\n"
                f"Required keys are: {', '.join(required_keys)}"
            )

        # Create arguments object
        try:
            arg = ARG(
                model=arg_dict["model"],
                data_source=arg_dict["data_source"],
                rating=arg_dict["rating"],
                diversity=arg_dict.get("diversity", False),
                coverage=arg_dict.get("coverage", False),
                coherence=arg_dict.get("coherence", False),
                repetitive=arg_dict.get("repetitive", False),
                readability=arg_dict.get("readability", False),
                clustering_complementarity=arg_dict.get("clustering_complementarity", False)
            )
        except TypeError as e:
            raise TypeError(
                f"Invalid argument types in config file: {str(e)}\n"
                "Please check if the config values match the expected types."
            )
        except Exception as e:
            raise RuntimeError(f"Error creating argument object: {str(e)}")

        # Log configuration
        print("\nConfiguration:")
        print(f"- Config file: {config_path}")
        print(f"- Arguments: {arg}")
        
        # Run main function
        try:
            main(arg)
        except Exception as e:
            raise RuntimeError(f"Error in main execution: {str(e)}")

    except Exception as e:
        print("\nError executing evaluation script:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        
        # Print full traceback for debugging
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Print system info for debugging
        import platform
        print("\nSystem information:")
        print(f"Python version: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        
        sys.exit(1)

