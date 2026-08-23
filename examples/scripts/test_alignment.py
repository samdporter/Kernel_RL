#!/usr/bin/env python3
"""Test script to verify alignment between guidance and emission images.

This script loads the mk-h001 PET and T1 images and creates several
visualizations to help determine if they are properly aligned:
  1. Side-by-side comparison (with and without flip)
  2. Overlay visualization with transparency
  3. Edge overlay to check anatomical correspondence
  4. Multiple slice views at different depths

Usage:
    python scripts/test_alignment.py
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from krl.utils import load_image

# Configuration
DATA_PATH = Path("data/MK-H001")
PET_FILE = "MK-H001_PET_MNI.nii"
T1_FILE = "MK-H001_T1_MNI.nii"
OUTPUT_DIR = Path("alignment_test_output")


def normalize_intensity(img_array):
    """Normalize image intensities to [0, 1] range."""
    img_min = np.min(img_array)
    img_max = np.max(img_array)
    if img_max == img_min:
        return np.zeros_like(img_array)
    return (img_array - img_min) / (img_max - img_min)


def create_edge_map(img_array, sigma=2.0):
    """Create simple edge map using gradient magnitude."""
    from scipy.ndimage import gaussian_filter, sobel

    # Smooth first to reduce noise
    smoothed = gaussian_filter(img_array, sigma=sigma)

    # Compute gradients
    grad_z = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    grad_x = sobel(smoothed, axis=2)

    # Gradient magnitude
    grad_mag = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)

    return normalize_intensity(grad_mag)


def visualize_alignment(pet_array, t1_array, t1_flipped_array, output_dir):
    """Create comprehensive alignment visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize intensities for better visualization
    pet_norm = normalize_intensity(pet_array)
    t1_norm = normalize_intensity(t1_array)
    t1_flip_norm = normalize_intensity(t1_flipped_array)

    # Select slices at different depths
    n_slices = pet_array.shape[0]
    slice_positions = [
        n_slices // 4,
        n_slices // 2,
        3 * n_slices // 4
    ]

    # 1. Side-by-side comparison (original T1)
    print("Creating side-by-side comparison (original T1)...")
    fig, axes = plt.subplots(2, len(slice_positions), figsize=(18, 6))
    fig.suptitle("PET vs T1 (Original - No Flip)", fontsize=16, fontweight='bold')

    for i, z in enumerate(slice_positions):
        axes[0, i].imshow(pet_norm[z], cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f"PET - Slice {z}/{n_slices}")
        axes[0, i].axis('off')

        axes[1, i].imshow(t1_norm[z], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"T1 (original) - Slice {z}/{n_slices}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "01_sidebyside_original.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Side-by-side comparison (flipped T1)
    print("Creating side-by-side comparison (flipped T1)...")
    fig, axes = plt.subplots(2, len(slice_positions), figsize=(18, 6))
    fig.suptitle("PET vs T1 (Flipped along Z-axis)", fontsize=16, fontweight='bold')

    for i, z in enumerate(slice_positions):
        axes[0, i].imshow(pet_norm[z], cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f"PET - Slice {z}/{n_slices}")
        axes[0, i].axis('off')

        axes[1, i].imshow(t1_flip_norm[z], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"T1 (flipped) - Slice {z}/{n_slices}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "02_sidebyside_flipped.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Overlay visualization (original T1)
    print("Creating overlay visualization (original T1)...")
    fig, axes = plt.subplots(1, len(slice_positions), figsize=(18, 6))
    if len(slice_positions) == 1:
        axes = [axes]
    fig.suptitle("Overlay: PET (red) + T1 Original (green)", fontsize=16, fontweight='bold')

    for i, z in enumerate(slice_positions):
        # Create RGB overlay: PET in red, T1 in green
        overlay = np.zeros((*pet_norm[z].shape, 3))
        overlay[..., 0] = pet_norm[z]  # Red channel = PET
        overlay[..., 1] = t1_norm[z]   # Green channel = T1

        axes[i].imshow(overlay)
        axes[i].set_title(f"Slice {z}/{n_slices} - Yellow = Good Overlap")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "03_overlay_original.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Overlay visualization (flipped T1)
    print("Creating overlay visualization (flipped T1)...")
    fig, axes = plt.subplots(1, len(slice_positions), figsize=(18, 6))
    if len(slice_positions) == 1:
        axes = [axes]
    fig.suptitle("Overlay: PET (red) + T1 Flipped (green)", fontsize=16, fontweight='bold')

    for i, z in enumerate(slice_positions):
        # Create RGB overlay: PET in red, T1 in green
        overlay = np.zeros((*pet_norm[z].shape, 3))
        overlay[..., 0] = pet_norm[z]      # Red channel = PET
        overlay[..., 1] = t1_flip_norm[z]  # Green channel = T1 flipped

        axes[i].imshow(overlay)
        axes[i].set_title(f"Slice {z}/{n_slices} - Yellow = Good Overlap")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "04_overlay_flipped.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Edge overlay comparison (original T1)
    print("Creating edge overlay (original T1)...")
    pet_edges = create_edge_map(pet_array)
    t1_edges = create_edge_map(t1_array)

    fig, axes = plt.subplots(1, len(slice_positions), figsize=(18, 6))
    if len(slice_positions) == 1:
        axes = [axes]
    fig.suptitle("Edge Overlay: PET edges (cyan) + T1 Original edges (magenta)",
                 fontsize=16, fontweight='bold')

    for i, z in enumerate(slice_positions):
        overlay = np.zeros((*pet_edges[z].shape, 3))
        overlay[..., 0] = t1_edges[z]  # Red channel = T1 edges
        overlay[..., 1] = pet_edges[z]  # Green channel = PET edges
        overlay[..., 2] = pet_edges[z]  # Blue channel = PET edges
        # Result: PET edges = cyan, T1 edges = magenta, overlap = white

        axes[i].imshow(overlay)
        axes[i].set_title(f"Slice {z}/{n_slices} - White = Edge Alignment")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "05_edges_original.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Edge overlay comparison (flipped T1)
    print("Creating edge overlay (flipped T1)...")
    t1_flip_edges = create_edge_map(t1_flipped_array)

    fig, axes = plt.subplots(1, len(slice_positions), figsize=(18, 6))
    if len(slice_positions) == 1:
        axes = [axes]
    fig.suptitle("Edge Overlay: PET edges (cyan) + T1 Flipped edges (magenta)",
                 fontsize=16, fontweight='bold')

    for i, z in enumerate(slice_positions):
        overlay = np.zeros((*pet_edges[z].shape, 3))
        overlay[..., 0] = t1_flip_edges[z]  # Red channel = T1 flipped edges
        overlay[..., 1] = pet_edges[z]      # Green channel = PET edges
        overlay[..., 2] = pet_edges[z]      # Blue channel = PET edges

        axes[i].imshow(overlay)
        axes[i].set_title(f"Slice {z}/{n_slices} - White = Edge Alignment")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "06_edges_flipped.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Checkerboard pattern comparison
    print("Creating checkerboard comparison...")
    fig, axes = plt.subplots(2, len(slice_positions), figsize=(18, 8))
    fig.suptitle("Checkerboard Pattern (helps spot misalignment)",
                 fontsize=16, fontweight='bold')

    checker_size = 16  # Size of checkerboard squares

    for i, z in enumerate(slice_positions):
        # Original T1
        checker_orig = np.zeros_like(pet_norm[z])
        for y in range(0, pet_norm.shape[1], checker_size):
            for x in range(0, pet_norm.shape[2], checker_size):
                if ((y // checker_size) + (x // checker_size)) % 2 == 0:
                    checker_orig[
                        y:min(y+checker_size, pet_norm.shape[1]),
                        x:min(x+checker_size, pet_norm.shape[2])
                    ] = pet_norm[z][
                        y:min(y+checker_size, pet_norm.shape[1]),
                        x:min(x+checker_size, pet_norm.shape[2])
                    ]
                else:
                    checker_orig[
                        y:min(y+checker_size, pet_norm.shape[1]),
                        x:min(x+checker_size, pet_norm.shape[2])
                    ] = t1_norm[z][
                        y:min(y+checker_size, pet_norm.shape[1]),
                        x:min(x+checker_size, pet_norm.shape[2])
                    ]

        axes[0, i].imshow(checker_orig, cmap='gray')
        axes[0, i].set_title(f"Slice {z} - Original T1")
        axes[0, i].axis('off')

        # Flipped T1
        checker_flip = np.zeros_like(pet_norm[z])
        for y in range(0, pet_norm.shape[1], checker_size):
            for x in range(0, pet_norm.shape[2], checker_size):
                if ((y // checker_size) + (x // checker_size)) % 2 == 0:
                    checker_flip[
                        y:min(y+checker_size, pet_norm.shape[1]),
                        x:min(x+checker_size, pet_norm.shape[2])
                    ] = pet_norm[z][
                        y:min(y+checker_size, pet_norm.shape[1]),
                        x:min(x+checker_size, pet_norm.shape[2])
                    ]
                else:
                    checker_flip[
                        y:min(y+checker_size, pet_norm.shape[1]),
                        x:min(x+checker_size, pet_norm.shape[2])
                    ] = t1_flip_norm[z][
                        y:min(y+checker_size, pet_norm.shape[1]),
                        x:min(x+checker_size, pet_norm.shape[2])
                    ]

        axes[1, i].imshow(checker_flip, cmap='gray')
        axes[1, i].set_title(f"Slice {z} - Flipped T1")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "07_checkerboard.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll visualizations saved to: {output_dir.resolve()}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("MK-H001 Alignment Test")
    print("=" * 70)
    print()

    # Load images
    pet_path = DATA_PATH / PET_FILE
    t1_path = DATA_PATH / T1_FILE

    print(f"Loading PET image from: {pet_path}")
    if not pet_path.exists():
        print(f"ERROR: PET file not found at {pet_path}")
        print("Please check the data path and file name.")
        return

    print(f"Loading T1 image from: {t1_path}")
    if not t1_path.exists():
        print(f"ERROR: T1 file not found at {t1_path}")
        print("Please check the data path and file name.")
        return

    pet_img = load_image(pet_path)
    t1_img = load_image(t1_path)

    pet_array = pet_img.as_array()
    t1_array = t1_img.as_array()

    print()
    print(f"PET shape: {pet_array.shape}")
    print(f"T1 shape:  {t1_array.shape}")
    print()

    if pet_array.shape != t1_array.shape:
        print("WARNING: PET and T1 images have different shapes!")
        print("They should be the same size if they're in the same space (MNI).")
        print()

    # Create flipped version
    t1_flipped_array = np.flip(t1_array, axis=0)

    print("Creating visualizations...")
    print("This will generate several PNG files to help you assess alignment:")
    print("  1. Side-by-side views (original T1)")
    print("  2. Side-by-side views (flipped T1)")
    print("  3. Color overlay (original T1) - yellow indicates overlap")
    print("  4. Color overlay (flipped T1) - yellow indicates overlap")
    print("  5. Edge overlay (original T1) - white indicates edge alignment")
    print("  6. Edge overlay (flipped T1) - white indicates edge alignment")
    print("  7. Checkerboard pattern - smooth transitions indicate alignment")
    print()

    visualize_alignment(pet_array, t1_array, t1_flipped_array, OUTPUT_DIR)

    print()
    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print()
    print("Look for these indicators of good alignment:")
    print()
    print("1. OVERLAY IMAGES (03, 04):")
    print("   - Yellow/white regions indicate spatial overlap")
    print("   - Brain structures should overlap (especially ventricles)")
    print()
    print("2. EDGE OVERLAY IMAGES (05, 06):")
    print("   - White lines indicate edges that align between PET and T1")
    print("   - Skull boundaries and ventricles are good landmarks")
    print()
    print("3. CHECKERBOARD IMAGES (07):")
    print("   - Smooth transitions at checkerboard boundaries = good alignment")
    print("   - Discontinuities or jumps = misalignment")
    print()
    print("4. Compare 'original' vs 'flipped' versions:")
    print("   - The version with better overlap is the correct orientation")
    print()
    print("Current configuration in run_deconv_sweeps.py:")
    print("   flip_guidance: True (line 114)")
    print()
    print("If the ORIGINAL T1 looks better aligned, change to:")
    print("   flip_guidance: False")
    print()
    print("If the FLIPPED T1 looks better aligned, keep:")
    print("   flip_guidance: True")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
