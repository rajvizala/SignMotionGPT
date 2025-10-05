"""
Standalone inference script for Motion LLM
"""
import argparse
import json
import torch
from typing import Optional, List
from data import (
    load_dataset, build_motion_vocab, compute_length_stats,
    build_prompt_vocab, motion_specials_to_ids
)
from model import get_motion_token_info
from generate import generate_t2m
from train import load_model_from_hub


class MotionGenerator:
    """
    Easy-to-use interface for motion generation
    """
    
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        device: str = "cuda:0"
    ):
        """
        Initialize generator
        
        Args:
            model_path: Path to trained model (local or HF hub)
            dataset_path: Path to training dataset JSON (for stats)
            device: Device to run inference on
        """
        print(f"Loading model from {model_path}...")
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_model_from_hub(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Load dataset for statistics
        print(f"Loading dataset from {dataset_path}...")
        self.dataset = load_dataset(dataset_path)
        
        # Build vocabulary and stats
        print("Building motion vocabulary and statistics...")
        codebook_size, _ = build_motion_vocab(self.dataset)
        self.length_stats_by_text, self.global_median_len = compute_length_stats(self.dataset)
        self.prompt_vocab = build_prompt_vocab(self.dataset)
        
        # Get motion token IDs
        self.motion_token_ids, self.mot_begin_id, self.mot_end_id = get_motion_token_info(
            self.tokenizer, codebook_size
        )
        
        # Check for participant IDs
        self.has_pid = "participant_id" in self.dataset.column_names
        
        print("✓ Generator ready!")
    
    def generate(
        self,
        prompt: str,
        participant_id: Optional[str] = None,
        max_new_tokens: int = 256,
        per_prompt_vocab: bool = True,
        return_tokens: bool = True
    ) -> str:
        """
        Generate motion from text prompt
        
        Args:
            prompt: Text description of motion
            participant_id: Optional participant ID for personalized generation
            max_new_tokens: Maximum tokens to generate
            per_prompt_vocab: Restrict to tokens seen with this prompt in training
            return_tokens: If True, return space-separated token IDs. If False, return full response
        
        Returns:
            Generated motion tokens (space-separated IDs) or full response
        """
        # Generate
        response = generate_t2m(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_text=prompt,
            mot_begin_id=self.mot_begin_id,
            mot_end_id=self.mot_end_id,
            motion_token_ids=self.motion_token_ids,
            length_stats_by_text=self.length_stats_by_text,
            global_median_len=self.global_median_len,
            prompt_vocab=self.prompt_vocab if per_prompt_vocab else None,
            pid=participant_id,
            has_pid=self.has_pid,
            max_new_tokens=max_new_tokens,
            per_prompt_vocab=per_prompt_vocab
        )
        
        if return_tokens:
            # Extract motion tokens
            span = response.split("<MOT_BEGIN>")[-1]
            span = span.split("<MOT_END>")[0]
            token_ids = motion_specials_to_ids(span)
            return " ".join(str(id) for id in token_ids)
        else:
            return response
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate motions for multiple prompts
        
        Args:
            prompts: List of text descriptions
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            List of generated motion token sequences
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate motions from text prompts")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (local directory or HF hub)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to training dataset JSON"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for motion generation"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="Path to text file with one prompt per line"
    )
    parser.add_argument(
        "--participant_id",
        type=str,
        default=None,
        help="Optional participant ID for personalized generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (default: print to stdout)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--no_prompt_vocab",
        action="store_true",
        help="Disable per-prompt vocabulary restriction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MotionGenerator(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        device=args.device
    )
    
    # Collect prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    
    if not prompts:
        print("Error: Please provide --prompt or --prompts_file")
        return
    
    # Generate
    print(f"\nGenerating motions for {len(prompts)} prompt(s)...\n")
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Prompt: {prompt}")
        
        motion_tokens = generator.generate(
            prompt=prompt,
            participant_id=args.participant_id,
            max_new_tokens=args.max_new_tokens,
            per_prompt_vocab=not args.no_prompt_vocab
        )
        
        print(f"Generated: {motion_tokens}\n")
        results.append({
            "prompt": prompt,
            "motion_tokens": motion_tokens
        })
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {args.output}")
    
    print("Done!")


if __name__ == "__main__":
    main()