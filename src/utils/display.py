"""Display utilities for image visualization and debugging."""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_image(image: Image.Image, 
               title: str = "Image", 
               figsize: Tuple[int, int] = (10, 8),
               save_path: str = None) -> None:
    """Display an image using matplotlib.
    
    Args:
        image: PIL Image to display
        title: Title for the display window
        figsize: Figure size as (width, height)
        save_path: Optional path to save the displayed image
    """
    plt.figure(figsize=figsize)
    
    if image.mode == '1':
        # Binary image
        plt.imshow(image, cmap='gray', interpolation='nearest')
    elif image.mode == 'L':
        # Grayscale image
        plt.imshow(image, cmap='gray')
    else:
        # Color image
        plt.imshow(image)
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logging.getLogger(__name__).info(f"Display saved to {save_path}")
    
    plt.show()


def display_processing_stages(stage_results: Dict[str, Image.Image],
                            save_dir: str = None) -> None:
    """Display all processing stages in a grid layout.
    
    Args:
        stage_results: Dictionary mapping stage names to PIL Images
        save_dir: Optional directory to save individual stage images
    """
    stages = list(stage_results.keys())
    num_stages = len(stages)
    
    if num_stages == 0:
        return
    
    # Calculate grid dimensions
    cols = min(3, num_stages)
    rows = (num_stages + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle('Image Processing Pipeline Stages', fontsize=16)
    
    # Handle case where we have only one subplot
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, stage_name in enumerate(stages):
        image = stage_results[stage_name]
        ax = axes[i]
        
        if image.mode == '1':
            ax.imshow(image, cmap='gray', interpolation='nearest')
        elif image.mode == 'L':
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        
        ax.set_title(stage_name.replace('_', ' ').title())
        ax.axis('off')
        
        # Save individual stage if requested
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"stage_{stage_name}.png")
            image.save(save_path)
    
    # Hide unused subplots
    for i in range(num_stages, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_comparison_grid(images: Dict[str, Image.Image],
                          title: str = "Image Comparison",
                          max_cols: int = 3) -> None:
    """Create a comparison grid of images.
    
    Args:
        images: Dictionary mapping labels to PIL Images
        title: Overall title for the grid
        max_cols: Maximum number of columns in the grid
    """
    labels = list(images.keys())
    num_images = len(labels)
    
    if num_images == 0:
        return
    
    cols = min(max_cols, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(title, fontsize=16)
    
    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, label in enumerate(labels):
        image = images[label]
        ax = axes[i]
        
        if image.mode == '1':
            ax.imshow(image, cmap='gray', interpolation='nearest')
        elif image.mode == 'L':
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        
        ax.set_title(label)
        ax.axis('off')
        
        # Add image info as text
        info_text = f"{image.size[0]}x{image.size[1]}\n{image.mode}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_histogram(image: Image.Image,
                  title: str = "Image Histogram",
                  bins: int = 256) -> None:
    """Display histogram of image pixel values.
    
    Args:
        image: PIL Image
        title: Title for the histogram
        bins: Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    
    if image.mode == 'RGB':
        # Color histogram
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            channel_data = np.array(image)[:, :, i].flatten()
            plt.hist(channel_data, bins=bins, alpha=0.7, color=color, label=color.upper())
        plt.legend()
    else:
        # Grayscale histogram
        pixel_values = np.array(image).flatten()
        plt.hist(pixel_values, bins=bins, color='gray', alpha=0.7)
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def show_before_after(before: Image.Image, 
                     after: Image.Image,
                     before_title: str = "Before",
                     after_title: str = "After",
                     overall_title: str = "Before and After Comparison") -> None:
    """Show before and after images side by side.
    
    Args:
        before: Original PIL Image
        after: Processed PIL Image
        before_title: Title for the before image
        after_title: Title for the after image
        overall_title: Overall comparison title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(overall_title, fontsize=16)
    
    # Before image
    if before.mode == '1':
        ax1.imshow(before, cmap='gray', interpolation='nearest')
    elif before.mode == 'L':
        ax1.imshow(before, cmap='gray')
    else:
        ax1.imshow(before)
    ax1.set_title(f"{before_title}\n{before.size[0]}x{before.size[1]} ({before.mode})")
    ax1.axis('off')
    
    # After image
    if after.mode == '1':
        ax2.imshow(after, cmap='gray', interpolation='nearest')
    elif after.mode == 'L':
        ax2.imshow(after, cmap='gray')
    else:
        ax2.imshow(after)
    ax2.set_title(f"{after_title}\n{after.size[0]}x{after.size[1]} ({after.mode})")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_dithering_comparison(original: Image.Image,
                                 dithered_results: Dict[str, Image.Image]) -> None:
    """Visualize comparison of different dithering algorithms.
    
    Args:
        original: Original grayscale image
        dithered_results: Dictionary mapping algorithm names to dithered images
    """
    num_results = len(dithered_results)
    cols = min(3, num_results + 1)  # +1 for original
    rows = (num_results + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle('Dithering Algorithm Comparison', fontsize=16)
    
    # Handle subplot array access
    if rows == 1:
        if cols == 1:
            axes = [axes]
        else:
            axes = axes
    else:
        axes = axes.flatten()
    
    # Show original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show dithered results
    for i, (algorithm, dithered) in enumerate(dithered_results.items(), 1):
        axes[i].imshow(dithered, cmap='gray', interpolation='nearest')
        axes[i].set_title(algorithm.replace('_', ' ').title())
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_results + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_crop_suggestion(image: Image.Image,
                        crop_box: Tuple[int, int, int, int],
                        title: str = "Suggested Crop Area") -> None:
    """Visualize a suggested crop area on an image.
    
    Args:
        image: PIL Image to show crop suggestion for
        crop_box: Crop box as (left, top, right, bottom)
        title: Title for the visualization
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Draw crop rectangle
    left, top, right, bottom = crop_box
    width = right - left
    height = bottom - top
    
    rect = patches.Rectangle((left, top), width, height, 
                           linewidth=3, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    
    # Add crop dimensions as text
    crop_text = f"Crop: {width}x{height}\nPosition: ({left}, {top})"
    plt.text(left + 10, top + 20, crop_text, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12, color='red')
    
    plt.title(title)
    plt.axis('off')
    plt.show()


def save_processing_report(stage_results: Dict[str, Image.Image],
                          timings: Dict[str, float],
                          output_path: str) -> None:
    """Save a comprehensive processing report with images and timings.
    
    Args:
        stage_results: Dictionary of processing stage results
        timings: Dictionary of processing timings
        output_path: Path to save the report
    """
    import os
    
    # Create output directory
    report_dir = os.path.dirname(output_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    
    # Save individual stage images
    for stage_name, image in stage_results.items():
        stage_filename = f"{os.path.splitext(output_path)[0]}_{stage_name}.png"
        image.save(stage_filename)
    
    # Create timing report
    timing_report = []
    timing_report.append("DotShot Processing Report")
    timing_report.append("=" * 40)
    timing_report.append("")
    timing_report.append("Stage Timings:")
    
    total_time = 0
    for stage_name, timing in timings.items():
        timing_report.append(f"  {stage_name}: {timing:.3f}s")
        total_time += timing
    
    timing_report.append("")
    timing_report.append(f"Total Processing Time: {total_time:.3f}s")
    timing_report.append("")
    timing_report.append("Stage Results:")
    
    for stage_name, image in stage_results.items():
        timing_report.append(f"  {stage_name}: {image.size[0]}x{image.size[1]} ({image.mode})")
    
    # Save timing report
    timing_filename = f"{os.path.splitext(output_path)[0]}_report.txt"
    with open(timing_filename, 'w') as f:
        f.write('\n'.join(timing_report))
    
    logging.getLogger(__name__).info(f"Processing report saved to {timing_filename}")
