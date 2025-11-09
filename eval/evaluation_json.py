#!/usr/bin/env python3
"""
Evaluate model responses and add correctness labels.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def evaluate_math_response(response: str, answer: str) -> bool:
    """
    Compare model response with ground truth answer.
    
    This is a simple implementation. You may need more sophisticated
    logic depending on your task (e.g., extracting answers from text,
    normalizing formats, etc.)
    """
    # Extract answer from response (customize based on your format)
    response_clean = response.strip().lower()
    answer_clean = answer.strip().lower()
    
    # Simple exact match (you should improve this)
    return response_clean == answer_clean or answer_clean in response_clean


def evaluate_file(input_path: Path, output_path: Path) -> None:
    """
    Read a JSONL file, evaluate each response, and write results with 'correct' column.
    """
    df = pd.read_json(input_path, lines=True)
    
    if df.empty:
        print(f"[warn] Empty file: {input_path}")
        return
    
    # Add correctness evaluation
    correct_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {input_path.name}"):
        is_correct = evaluate_math_response(row['response'], row['answer'])
        correct_list.append(1 if is_correct else 0)
    
    df['correct'] = correct_list
    
    # Write to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient='records', lines=True)
    
    accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0
    print(f"[done] {input_path.name}: accuracy = {accuracy:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses")
    parser.add_argument("--root_path", type=str, required=True,
                        help="Root directory containing results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()
    
    root_path = Path(args.root_path)
    output_dir = Path(args.output_dir) if args.output_dir else root_path
    
    # Find all JSONL files
    for jsonl_file in root_path.rglob("*.jsonl"):
        if "raw" in jsonl_file.name:
            continue
        
        # Create corresponding output path
        rel_path = jsonl_file.relative_to(root_path)
        output_file = output_dir / rel_path
        
        print(f"\nProcessing: {jsonl_file}")
        evaluate_file(jsonl_file, output_file)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()