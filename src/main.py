#!/usr/bin/env python3
"""Main application entry point for DotShot."""

import os
import sys
import time
import logging
import argparse
from typing import Optional
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import project modules
from config.settings import SETTINGS
from camera import CameraController
from processing import ProcessingPipeline, DitheringProcessor
from printer import PrinterController
from photobooth import PhotoboothController
from utils.display import (show_image, display_processing_stages, 
                          create_comparison_grid, visualize_dithering_comparison)
from utils.testing import TestImageLoader, validate_pipeline


def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        SETTINGS["system"].OUTPUT_DIR,
        SETTINGS["system"].TEMP_DIR,
        SETTINGS["system"].TEST_IMAGES_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def setup_global_options(args: argparse.Namespace) -> None:
    """Setup global options like debug and verbose mode."""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        SETTINGS["system"].DEBUG_MODE = True
        print("Debug mode enabled")
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("Verbose mode enabled")
    
    setup_directories()


def cmd_preview(args: argparse.Namespace) -> None:
    """Show camera preview."""
    print("Starting camera preview...")
    
    if args.mock_camera:
        from camera.capture import MockCameraController
        camera = MockCameraController()
    else:
        camera = CameraController()
    
    if camera.initialize():
        camera.preview_image(args.preview_time)
        print("Preview completed")
    else:
        print("Failed to initialize camera", file=sys.stderr)
        sys.exit(1)
    
    camera.cleanup()


def cmd_capture(args: argparse.Namespace) -> None:
    """Capture image from camera and optionally print it."""
    print("Initializing camera...")
    
    if args.mock_camera:
        from camera.capture import MockCameraController
        camera = MockCameraController()
    else:
        camera = CameraController()
    
    if not camera.initialize():
        print("Failed to initialize camera", file=sys.stderr)
        sys.exit(1)
    
    # Capture image
    print("Capturing image...")
    image = camera.capture_image(args.output)
    camera.cleanup()
    
    if not image:
        print("Failed to capture image", file=sys.stderr)
        sys.exit(1)
    
    print(f"Image captured: {image.size[0]}x{image.size[1]}")
    
    # Process image
    print("Processing image...")
    pipeline = ProcessingPipeline()
    processed_image = pipeline.process_image(image, display_stages=args.display_stages)
    
    if args.display_stages:
        display_processing_stages(pipeline.get_all_stage_results())
    
    # Apply dithering
    print("Applying dithering...")
    dithering = DitheringProcessor()
    dithered_image = dithering.apply_dithering(processed_image)
    
    # Show final result
    show_image(dithered_image, title="Final Processed Image")
    
    if not args.preview_only:
        # Print the image
        print("Connecting to printer...")
        printer = PrinterController()
        
        if printer.connect():
            print("Printing image...")
            if printer.print_image(dithered_image, caption=args.text):
                print("Print job completed successfully!")
            else:
                print("Print job failed", file=sys.stderr)
            printer.disconnect()
        else:
            print("Failed to connect to printer", file=sys.stderr)
    
    # Save processed image
    processed_filename = f"processed_{args.output}"
    dithered_image.save(processed_filename)
    print(f"Processed image saved as {processed_filename}")


def cmd_process_file(args: argparse.Namespace) -> None:
    """Process an image file through the pipeline."""
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading image: {args.input_file}")
    
    try:
        image = Image.open(args.input_file)
    except Exception as e:
        print(f"Failed to load image: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Image loaded: {image.size[0]}x{image.size[1]} ({image.mode})")
    
    # Process through pipeline
    print("Processing image through pipeline...")
    pipeline = ProcessingPipeline()
    processed_image = pipeline.process_image(
        image, 
        save_intermediates=args.save_stages,
        display_stages=args.display_stages
    )
    
    if args.display_stages:
        display_processing_stages(pipeline.get_all_stage_results())
    
    # Show timing information
    timings = pipeline.get_stage_timings()
    print("\nProcessing times:")
    for stage, time_taken in timings.items():
        print(f"  {stage}: {time_taken:.3f}s")
    
    # Dithering
    dithering = DitheringProcessor()
    
    if args.compare_dithering:
        print("Comparing dithering algorithms...")
        dithering_results = dithering.compare_algorithms(processed_image)
        visualize_dithering_comparison(processed_image.convert('L'), dithering_results)
        
        # Find optimal settings
        optimal_image, optimal_settings = dithering.optimize_for_printer(processed_image)
        print(f"Optimal dithering: {optimal_settings}")
        show_image(optimal_image, title="Optimized for Printing")
        final_image = optimal_image
    else:
        final_image = dithering.apply_dithering(processed_image)
        show_image(final_image, title="Dithered Image")
    
    # Save output
    if args.output:
        output_filename = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_filename = f"processed_{base_name}.png"
    
    final_image.save(output_filename)
    print(f"Processed image saved as {output_filename}")


def cmd_test_printer(args: argparse.Namespace) -> None:
    """Test printer connection and functionality."""
    print("Testing printer connection...")
    
    printer = PrinterController()
    
    if printer.connect(args.device_path):
        print("✓ Printer connected successfully")
        
        # Get printer info
        info = printer.get_printer_info()
        print(f"✓ Device path: {info['device_path']}")
        print(f"✓ Configuration: {info['settings']}")
        
        # Test print
        response = input("Would you like to run a test print? (y/N): ")
        if response.lower() in ['y', 'yes']:
            print("Sending test print...")
            if printer.test_connection():
                print("✓ Test print completed successfully!")
            else:
                print("✗ Test print failed", file=sys.stderr)
        
        printer.disconnect()
    else:
        print("✗ Failed to connect to printer", file=sys.stderr)
        
        # Show available devices
        print("\nChecking for available devices:")
        device_paths = [SETTINGS["printer"].DEVICE_PATH] + SETTINGS["printer"].BACKUP_DEVICE_PATHS
        
        for path in device_paths:
            if os.path.exists(path):
                print(f"  ✓ Found: {path}")
            else:
                print(f"  ✗ Not found: {path}")


def cmd_test_pipeline(args: argparse.Namespace) -> None:
    """Test the image processing pipeline with test images."""
    print("Testing image processing pipeline...")
    
    # Setup test images
    test_loader = TestImageLoader()
    
    if args.generate_images:
        test_loader.generate_test_images()
    
    test_image_names = test_loader.list_test_images()
    
    if not test_image_names:
        print("No test images found. Use --generate-images to create them.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(test_image_names)} test images")
    
    # Load test images
    test_images = []
    for name in test_image_names[:3]:  # Test with first 3 images
        image = test_loader.load_image(name)
        if image:
            test_images.append(image)
            print(f"  ✓ Loaded: {name}")
        else:
            print(f"  ✗ Failed to load: {name}")
    
    if not test_images:
        print("No test images could be loaded", file=sys.stderr)
        sys.exit(1)
    
    # Validate pipeline
    print("Running pipeline validation...")
    pipeline = ProcessingPipeline()
    results = validate_pipeline(pipeline, test_images)
    
    # Display results
    print("\nValidation Results:")
    for test_name, result in results.items():
        if 'error' in result:
            print(f"  ✗ {test_name}: {result['error']}")
        else:
            times = result.get('processing_times', {})
            total_time = sum(times.values())
            print(f"  ✓ {test_name}: {total_time:.3f}s total")


def cmd_print_image(args: argparse.Namespace) -> None:
    """Print an image file directly (assumes it's already processed and dithered)."""
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        image = Image.open(args.input_file)
    except Exception as e:
        print(f"Failed to load image: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded image: {image.size[0]}x{image.size[1]} ({image.mode})")
    
    # Connect to printer
    printer = PrinterController()
    if not printer.connect(args.device_path):
        print("Failed to connect to printer", file=sys.stderr)
        sys.exit(1)
    
    # Print image
    print("Printing image...")
    if printer.print_image(image, caption=args.text):
        print("✓ Print job completed successfully!")
    else:
        print("✗ Print job failed", file=sys.stderr)
    
    printer.disconnect()


def cmd_photobooth(args: argparse.Namespace) -> None:
    """Run the photobooth in automated mode."""
    print(f"🎉 Starting {SETTINGS['photobooth'].BOOTH_NAME}...")
    
    # Initialize photobooth controller
    photobooth = PhotoboothController()
    
    if not photobooth.initialize():
        print("❌ Failed to initialize photobooth", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Display initial status
        status = photobooth.get_status()
        print(f"📷 Camera: {'✓' if status['camera_ready'] else '✗'}")
        print(f"🖨️  Printer: {'✓' if status['printer_connected'] else '✗'}")
        print(f"🔘 GPIO: {'✓' if status['gpio_available'] else '✗ (Mock mode)'}")
        print()
        
        # Start photobooth operation
        photobooth.start()
        
        if args.manual_trigger:
            # Manual trigger mode for testing
            print("Manual trigger mode - press Enter to take photo, 'q' to quit")
            while True:
                user_input = input().strip().lower()
                if user_input == 'q':
                    break
                elif user_input == 'status':
                    status = photobooth.get_status()
                    print(f"Photos taken: {status['total_photos_taken']}")
                    print(f"Session active: {status['session_active']}")
                else:
                    # Trigger photo session manually
                    photobooth._on_button_press()
        else:
            # Continuous operation - wait for button presses
            print("Photobooth running... Press Ctrl+C to stop")
            while photobooth.running:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down photobooth...")
    except Exception as e:
        print(f"❌ Error in photobooth operation: {e}", file=sys.stderr)
    finally:
        photobooth.cleanup()
        print("✅ Photobooth shutdown complete")


def cmd_info(args: argparse.Namespace) -> None:
    """Display system information and configuration."""
    print("DotShot System Information")
    print("=" * 30)
    
    # Camera info
    print("\nCamera Configuration:")
    camera_settings = SETTINGS["camera"]
    print(f"  Resolution: {camera_settings.RESOLUTION}")
    print(f"  Exposure Mode: {camera_settings.EXPOSURE_MODE}")
    print(f"  AWB Mode: {camera_settings.AWB_MODE}")
    print(f"  ISO: {camera_settings.ISO}")
    
    # Processing info
    print("\nProcessing Configuration:")
    processing_settings = SETTINGS["processing"]
    print(f"  Target Size: {processing_settings.PRINTER_WIDTH_PIXELS}x{processing_settings.PRINTER_HEIGHT_PIXELS}")
    print(f"  Resize Algorithm: {processing_settings.RESIZE_ALGORITHM}")
    print(f"  Brightness Adjust: {processing_settings.BRIGHTNESS_ADJUST}")
    print(f"  Contrast Adjust: {processing_settings.CONTRAST_ADJUST}")
    print(f"  Edge Enhance Strength: {processing_settings.EDGE_ENHANCE_STRENGTH}")
    print(f"  Dither Algorithm: {processing_settings.DITHER_ALGORITHM}")
    
    # Printer info
    print("\nPrinter Configuration:")
    printer_settings = SETTINGS["printer"]
    print(f"  Device Path: {printer_settings.DEVICE_PATH}")
    print(f"  Print Density: {printer_settings.PRINT_DENSITY}")
    print(f"  Characters Per Line: {printer_settings.CHARS_PER_LINE}")
    print(f"  Lines Per Inch: {printer_settings.LINES_PER_INCH}")
    
    # Photobooth info
    print("\nPhotobooth Configuration:")
    photobooth_settings = SETTINGS["photobooth"]
    print(f"  Button Pin: GPIO {photobooth_settings.BUTTON_PIN}")
    print(f"  LED Pin: GPIO {photobooth_settings.LED_PIN}")
    print(f"  Countdown Seconds: {photobooth_settings.COUNTDOWN_SECONDS}")
    print(f"  Photos Per Session: {photobooth_settings.PHOTOS_PER_SESSION}")
    print(f"  Auto Print: {photobooth_settings.AUTO_PRINT}")
    print(f"  Booth Name: {photobooth_settings.BOOTH_NAME}")
    
    # Check camera availability
    print("\nSystem Status:")
    try:
        camera = CameraController()
        camera_available = camera.initialize()
        if camera_available:
            print("  ✓ Camera: Available")
            camera.cleanup()
        else:
            print("  ✗ Camera: Not available")
    except Exception as e:
        print(f"  ✗ Camera: Error ({e})")
    
    # Check printer paths
    printer = PrinterController()
    if printer.connect():
        print("  ✓ Printer: Available")
        printer.disconnect()
    else:
        print("  ✗ Printer: Not available")
    
    # Check GPIO
    try:
        import RPi.GPIO
        print("  ✓ GPIO: Available")
    except ImportError:
        print("  ✗ GPIO: Not available (will use mock mode)")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='DotShot - Automated Photo Booth System for Raspberry Pi'
    )
    
    # Global options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Show camera preview')
    preview_parser.add_argument('--preview-time', type=float, default=5.0,
                               help='Preview duration in seconds')
    preview_parser.add_argument('--mock-camera', action='store_true',
                               help='Use mock camera for testing')
    preview_parser.set_defaults(func=cmd_preview)
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture image from camera and optionally print it')
    capture_parser.add_argument('--output', '-o', default='captured_image.jpg',
                               help='Output filename')
    capture_parser.add_argument('--preview-only', action='store_true',
                               help='Only process and preview, don\'t print')
    capture_parser.add_argument('--display-stages', action='store_true',
                               help='Display intermediate processing stages')
    capture_parser.add_argument('--text', help='Text caption to print below image')
    capture_parser.add_argument('--mock-camera', action='store_true',
                               help='Use mock camera for testing')
    capture_parser.set_defaults(func=cmd_capture)
    
    # Process-file command
    process_parser = subparsers.add_parser('process-file', help='Process an image file through the pipeline')
    process_parser.add_argument('input_file', help='Input image file path')
    process_parser.add_argument('--display-stages', action='store_true',
                               help='Display intermediate processing stages')
    process_parser.add_argument('--save-stages', action='store_true',
                               help='Save intermediate processing stages')
    process_parser.add_argument('--compare-dithering', action='store_true',
                               help='Compare different dithering algorithms')
    process_parser.add_argument('--output', '-o', help='Output filename for processed image')
    process_parser.set_defaults(func=cmd_process_file)
    
    # Test-printer command
    test_printer_parser = subparsers.add_parser('test-printer', help='Test printer connection and functionality')
    test_printer_parser.add_argument('--device-path', help='Specific printer device path to test')
    test_printer_parser.set_defaults(func=cmd_test_printer)
    
    # Test-pipeline command
    test_pipeline_parser = subparsers.add_parser('test-pipeline', help='Test the image processing pipeline with test images')
    test_pipeline_parser.add_argument('--generate-images', action='store_true',
                                     help='Generate test images if needed')
    test_pipeline_parser.set_defaults(func=cmd_test_pipeline)
    
    # Print-image command
    print_parser = subparsers.add_parser('print-image', help='Print an image file directly (assumes it\'s already processed and dithered)')
    print_parser.add_argument('input_file', help='Input image file path')
    print_parser.add_argument('--text', help='Text caption to print below image')
    print_parser.add_argument('--device-path', help='Specific printer device path')
    print_parser.set_defaults(func=cmd_print_image)
    
    # Photobooth command
    photobooth_parser = subparsers.add_parser('photobooth', help='Run photobooth mode with automated capture and printing')
    photobooth_parser.add_argument('--manual-trigger', action='store_true',
                                   help='Enable manual trigger mode for testing (press Enter to capture)')
    photobooth_parser.set_defaults(func=cmd_photobooth)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display system information and configuration')
    info_parser.set_defaults(func=cmd_info)
    
    return parser


def main() -> None:
    """Main application entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup global options
    setup_global_options(args)
    
    # Call the appropriate command function
    args.func(args)


if __name__ == '__main__':
    main()
