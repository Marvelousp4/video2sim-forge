#!/usr/bin/env python3
"""
Step 2: SAM3 Segmentation

Generates segmentation masks for each object using SAM3 (Segment Anything Model 3).
This script MUST be run in the `sam3` conda environment.

Environment: sam3 (conda activate sam3)
Required: SAM3 model weights and sam3 package

Input:
    - RGB image (first frame from video)
    - prompts.txt (one object description per line)
    
Output:
    - mask_000.png, mask_001.png, ... (binary masks)
    - mask_to_prompt_mapping.json
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add SAM3 to path when it is installed from a local checkout.
SAM3_ROOT = os.environ.get("SAM3_ROOT", "")
if SAM3_ROOT:
    sys.path.append(SAM3_ROOT)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"ERROR: Cannot import SAM3. Make sure you're in the 'sam3' conda environment.")
    print(f"Run: conda activate sam3")
    print(f"Import error: {e}")
    sys.exit(1)


# def generate_masks(image_path: str, prompts: list[str], output_dir: str) -> list[str]:
#     """
#     Run SAM3 segmentation and save one mask per prompt.
    
#     Args:
#         image_path: Path to input RGB image
#         prompts: List of text prompts for segmentation
#         output_dir: Directory to save mask files
        
#     Returns:
#         List of mask file paths
#     """

def generate_masks(
    image_path: str,
    prompts: list[tuple[str, str]],
    output_dir: str,
) -> list[str]:
    """
    Run SAM3 segmentation and save one mask per prompt.

    Args:
        image_path: Path to input RGB image
        prompts: List of (prompt, fallback_prompt) tuples. If the primary
            prompt returns zero instances, the fallback is tried automatically.
            Pass fallback="" (or the same string) to disable the retry for an item.
        output_dir: Directory to save mask files

    Returns:
        List of mask file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup SAM3 with optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    
    print("Loading SAM3 model...")
    bpe_path = os.path.join(SAM3_ROOT, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    
    if os.path.exists(bpe_path):
        model = build_sam3_image_model(bpe_path=bpe_path)
    else:
        model = build_sam3_image_model()
    
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    
    # Initialize processor with confidence threshold
    processor = Sam3Processor(model, confidence_threshold=0.15)
    inference_state = processor.set_image(image)
    
    mask_index = 0
    mask_to_prompt_mapping = {}
    mask_paths = []
    
    # for prompt_idx, prompt in enumerate(prompts):
    #     print(f"\nProcessing prompt {prompt_idx}: '{prompt}'")
    #     processor.reset_all_prompts(inference_state)
        
    #     try:
    #         output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    #         masks = output.get("masks", [])
    #         scores = output.get("scores", [])
            
    #         if len(masks) == 0:
    #             print(f"  WARNING: No instances found for '{prompt}'")
    #             # Create empty mask as placeholder
    #             mask_uint8 = np.zeros((image.height, image.width), dtype=np.uint8)
    #         else:
    #             # Select best mask (highest confidence score)
    #             print(f"  Found {len(masks)} instance(s); selecting best")
                
    #             scores_f: list[float] = []
    #             for s in scores:
    #                 if isinstance(s, torch.Tensor):
    #                     s = s.cpu().item()
    #                 try:
    #                     scores_f.append(float(s))
    #                 except Exception:
    #                     scores_f.append(float("-inf"))

    #             best_i = int(np.argmax(scores_f)) if scores_f else 0
    #             best_score = scores_f[best_i] if scores_f else float("nan")
    #             best_mask = masks[best_i]
                
    #             if isinstance(best_mask, torch.Tensor):
    #                 best_mask = best_mask.cpu().numpy()

    #             if getattr(best_mask, "ndim", 0) > 2:
    #                 best_mask = best_mask.squeeze()

    #             # Convert to uint8 (0 or 255)
    #             mask_uint8 = (best_mask * 255).astype(np.uint8)
                
    #             if int(mask_uint8.max()) == 0:
    #                 print("  WARNING: Best mask is empty; writing empty placeholder")
    #                 mask_uint8 = np.zeros((image.height, image.width), dtype=np.uint8)
    #             else:
    #                 print(f"  Selected instance {best_i} with confidence: {best_score:.3f}")
            
    #     except Exception as e:
    #         print(f"  ERROR processing '{prompt}': {e}")
    #         mask_uint8 = np.zeros((image.height, image.width), dtype=np.uint8)
        
    #     # Save mask
    #     mask_filename = f"mask_{mask_index:03d}.png"
    #     mask_path = os.path.join(output_dir, mask_filename)
    #     Image.fromarray(mask_uint8).save(mask_path)
    #     print(f"  Saved: {mask_filename}")
        
    #     mask_to_prompt_mapping[mask_index] = prompt_idx
    #     mask_paths.append(mask_path)
    #     mask_index += 1


    def _try_prompt(prompt_text: str) -> tuple[np.ndarray | None, float, int]:
        """
        Run SAM3 on a single text prompt.

        Returns:
            (best_mask_uint8_or_None, best_score, num_instances).
            best_mask is None if no usable mask was produced.
        """
        processor.reset_all_prompts(inference_state)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
        masks = output.get("masks", [])
        scores = output.get("scores", [])

        if len(masks) == 0:
            return None, float("nan"), 0

        scores_f: list[float] = []
        for s in scores:
            if isinstance(s, torch.Tensor):
                s = s.cpu().item()
            try:
                scores_f.append(float(s))
            except Exception:
                scores_f.append(float("-inf"))

        best_i = int(np.argmax(scores_f)) if scores_f else 0
        best_score = scores_f[best_i] if scores_f else float("nan")
        best_mask = masks[best_i]

        if isinstance(best_mask, torch.Tensor):
            best_mask = best_mask.cpu().numpy()
        if getattr(best_mask, "ndim", 0) > 2:
            best_mask = best_mask.squeeze()

        mask_uint8 = (best_mask * 255).astype(np.uint8)
        if int(mask_uint8.max()) == 0:
            return None, best_score, len(masks)

        return mask_uint8, best_score, len(masks)

    for prompt_idx, (prompt, fallback) in enumerate(prompts):
        print(f"\nProcessing prompt {prompt_idx}: '{prompt}'")
        mask_uint8: np.ndarray | None = None
        used_prompt = prompt

        try:
            mask_uint8, best_score, n_inst = _try_prompt(prompt)
            if mask_uint8 is not None:
                print(f"  Found {n_inst} instance(s); selecting best")
                print(f"  Selected best with confidence: {best_score:.3f}")
            else:
                print(f"  No usable mask for '{prompt}' (instances={n_inst})")

                # # Retry with fallback if available and different
                # if fallback and fallback.strip() and fallback.strip() != prompt.strip():
                #     print(f"  Retrying with fallback prompt: '{fallback}'")
                #     try:
                #         mask_uint8, fb_score, fb_n = _try_prompt(fallback)
                #         if mask_uint8 is not None:
                #             used_prompt = fallback
                #             print(f"  Fallback found {fb_n} instance(s)")
                #             print(f"  Selected best with confidence: {fb_score:.3f}")
                #         else:
                #             print(f"  Fallback also produced no usable mask (instances={fb_n})")
                #     except Exception as e:
                #         print(f"  ERROR running fallback '{fallback}': {e}")
                # Build a cascade of fallback candidates to try, in order:
                #   1. The explicit fallback_prompt from Gemini
                #   2. The prompt with color/spatial words stripped (e.g. "blue chips can on right" -> "chips can")
                #   3. The head noun alone (e.g. "can")
                #   4. Common shape-synonyms for the head noun (e.g. "container", "jar")
                COLOR_WORDS = {
                    "red", "orange", "yellow", "green", "blue", "navy", "teal",
                    "cyan", "purple", "violet", "pink", "magenta", "brown",
                    "tan", "beige", "black", "white", "gray", "grey", "gold",
                    "silver", "clear", "transparent", "light", "dark", "bright",
                    "pale", "deep",
                }
                LOCATION_WORDS = {
                    "on", "in", "at", "near", "left", "right", "center",
                    "centre", "top", "bottom", "middle", "front", "back",
                    "edge", "side", "the", "of", "a", "an",
                }
                # Common interchangeable head nouns. SAM3 sometimes prefers one over another.
                SHAPE_SYNONYMS = {
                    "bottle": ["jar", "container", "shaker"],
                    "jar":    ["bottle", "container", "shaker"],
                    "shaker": ["jar", "bottle", "container"],
                    "can":    ["container", "cylinder", "tube"],
                    "container": ["bottle", "jar", "box"],
                    "box":    ["container", "package"],
                    "cup":    ["mug", "glass"],
                    "mug":    ["cup", "glass"],
                    "bowl":   ["dish", "plate"],
                    "plate":  ["dish", "tray"],
                }

                tokens = prompt.lower().split()
                core_no_color = [t for t in tokens if t not in COLOR_WORDS]
                core_no_loc = [t for t in core_no_color if t not in LOCATION_WORDS]

                candidates: list[str] = []

                def _add(c: str) -> None:
                    c = c.strip()
                    if c and c != prompt.strip() and c not in candidates:
                        candidates.append(c)

                # 1. Gemini-supplied fallback
                if fallback:
                    _add(fallback)
                # 2. Color/location stripped
                if core_no_loc:
                    _add(" ".join(core_no_loc))
                # 3. Head noun only
                head = core_no_loc[-1] if core_no_loc else ""
                if head:
                    _add(head)
                # 4. Shape synonyms applied to the head noun
                for alt in SHAPE_SYNONYMS.get(head, []):
                    # Replace head noun in the stripped phrase
                    if core_no_loc:
                        alt_phrase = " ".join(core_no_loc[:-1] + [alt])
                        _add(alt_phrase)
                    _add(alt)

                if not candidates:
                    print(f"  No fallback candidates to try.")
                else:
                    print(f"  Trying {len(candidates)} fallback candidate(s): {candidates}")

                for cand in candidates:
                    print(f"  Retrying with: '{cand}'")
                    try:
                        m, sc, nn = _try_prompt(cand)
                        if m is not None:
                            mask_uint8 = m
                            used_prompt = cand
                            print(f"  Retry '{cand}' found {nn} instance(s); confidence {sc:.3f}")
                            break  # Stop at first successful candidate
                        else:
                            print(f"  Retry '{cand}' produced no usable mask (instances={nn})")
                    except Exception as e:
                        print(f"  ERROR running retry '{cand}': {e}")

                

        except Exception as e:
            print(f"  ERROR processing '{prompt}': {e}")
            mask_uint8 = None

        if mask_uint8 is None:
            print(f"  WARNING: No instances found for '{prompt}' (fallback='{fallback}')")
            mask_uint8 = np.zeros((image.height, image.width), dtype=np.uint8)

        # Save mask
        mask_filename = f"mask_{mask_index:03d}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        Image.fromarray(mask_uint8).save(mask_path)
        print(f"  Saved: {mask_filename}  (prompt used: '{used_prompt}')")

        mask_to_prompt_mapping[mask_index] = {
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "fallback_prompt": fallback,
            "prompt_used": used_prompt,
        }
        mask_paths.append(mask_path)
        mask_index += 1
    
    # Save mapping file
    mapping_file = os.path.join(output_dir, "mask_to_prompt_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(mask_to_prompt_mapping, f, indent=2)
    
    print(f"\nTotal masks created: {mask_index} (prompts: {len(prompts)})")
    print(f"Mapping saved to: {mapping_file}")
    
    return mask_paths


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: SAM3 Segmentation - Generate masks from text prompts"
    )
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--prompts", required=True, help="Path to prompts file (one per line) or JSON file with 'objects' array")
    parser.add_argument("--output_dir", required=True, help="Directory to save masks")
    args = parser.parse_args()
    
    print("=" * 80)
    print("STEP 2: SAM3 Segmentation")
    print("=" * 80)
    print(f"Image: {args.image}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {args.output_dir}")
    
    # # Load prompts - support both plain text and JSON formats
    # prompts_path = Path(args.prompts)
    # if prompts_path.suffix == ".json":
    #     # Load from Gemini JSON output
    #     data = json.loads(prompts_path.read_text())
    #     if "objects" in data:
    #         prompts = [obj.get("prompt", "") for obj in data["objects"] if obj.get("prompt")]
    #     else:
    #         raise ValueError(f"JSON file must have 'objects' array: {args.prompts}")
    # else:
    #     # Load from plain text file
    #     with open(args.prompts, 'r') as f:
    #         prompts = [line.strip() for line in f if line.strip()]
    
    # if not prompts:
    #     raise ValueError(f"No prompts found in: {args.prompts}")
    
    # print(f"\nLoaded {len(prompts)} prompts:")
    # for i, p in enumerate(prompts):
    #     print(f"  {i}: {p}")

    # Load prompts - support both plain text and JSON formats.
    # Internal representation is a list of (prompt, fallback_prompt) tuples.
    prompts_path = Path(args.prompts)
    if prompts_path.suffix == ".json":
        # Load from Gemini JSON output
        data = json.loads(prompts_path.read_text())
        if "objects" not in data:
            raise ValueError(f"JSON file must have 'objects' array: {args.prompts}")
        prompts: list[tuple[str, str]] = []
        for obj in data["objects"]:
            p = (obj.get("prompt") or "").strip()
            if not p:
                continue
            fb = (obj.get("fallback_prompt") or "").strip()
            prompts.append((p, fb))
    else:
        # Plain text: one prompt per line. No fallback available in this format.
        with open(args.prompts, "r") as f:
            prompts = [(line.strip(), "") for line in f if line.strip()]

    if not prompts:
        raise ValueError(f"No prompts found in: {args.prompts}")

    print(f"\nLoaded {len(prompts)} prompts:")
    for i, (p, fb) in enumerate(prompts):
        if fb and fb != p:
            print(f"  {i}: {p}   (fallback: {fb})")
        else:
            print(f"  {i}: {p}")
    
    # Generate masks
    print("\n" + "-" * 40)
    mask_paths = generate_masks(args.image, prompts, args.output_dir)
    
    print("\n" + "=" * 80)
    print("STEP 2 COMPLETE")
    print("=" * 80)
    print(f"Generated {len(mask_paths)} masks in: {args.output_dir}")


if __name__ == "__main__":
    main()
