"""
Synthetic URL Testing - Command Line Interface
===============================================

CLI tool for testing phishing detection models with synthetic URLs.

USAGE EXAMPLES:
---------------
# Generate and test 1000 synthetic URLs
python ml/test_synthetic_urls.py --count 1000

# Test specific categories
python ml/test_synthetic_urls.py --categories banking ecommerce --count 500

# High realism mode
python ml/test_synthetic_urls.py --realism high --count 2000

# Export results
python ml/test_synthetic_urls.py --count 1000 --export results.json

# Show error analysis
python ml/test_synthetic_urls.py --count 500 --analyze-errors

ACADEMIC PURPOSE:
-----------------
This tool is designed for educational cybersecurity research using
ethically-generated synthetic phishing URLs with safe TLDs only.

Author: BCA Final Year Project - Phishing Detection System
License: Educational Use Only
"""

import argparse
import sys
import os
from typing import List, Optional

# Import synthetic URL modules
try:
    from ml.synthetic_url_generator import (
        SyntheticURLGenerator, RealismLevel, BrandCategory
    )
    from ml.synthetic_url_evaluator import SyntheticURLEvaluator
except ImportError:
    from synthetic_url_generator import (
        SyntheticURLGenerator, RealismLevel, BrandCategory
    )
    from synthetic_url_evaluator import SyntheticURLEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test phishing detection model with synthetic URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --count 1000
  %(prog)s --categories banking ecommerce --count 500
  %(prog)s --realism high --count 2000 --export results.json
  %(prog)s --count 500 --analyze-errors --max-errors 20
        """
    )
    
    # Generation options
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=1000,
        help='Total number of URLs to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--realism', '-r',
        type=str,
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Realism level for phishing URLs (default: medium)'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        choices=['banking', 'ecommerce', 'social', 'cloud', 'tech'],
        help='Brand categories to test (default: all)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Evaluation options
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model (default: ml/phishing_model.pkl)'
    )
    
    parser.add_argument(
        '--analyze-errors',
        action='store_true',
        help='Show detailed error analysis'
    )
    
    parser.add_argument(
        '--max-errors',
        type=int,
        default=10,
        help='Maximum errors to display in analysis (default: 10)'
    )
    
    # Export options
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to file (JSON or CSV based on extension)'
    )
    
    parser.add_argument(
        '--include-urls',
        action='store_true',
        help='Include full URL list in export'
    )
    
    # Display options
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    parser.add_argument(
        '--show-samples',
        type=int,
        default=0,
        help='Show N sample URLs before evaluation'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Print header
    if not args.quiet:
        print("=" * 70)
        print("SYNTHETIC PHISHING URL TESTING")
        print("=" * 70)
        print("\nETHICAL NOTICE:")
        print("All synthetic URLs use safe TLDs (.test, .example, .invalid)")
        print("These domains are reserved and will never be registered.")
        print("This tool is for educational cybersecurity research only.\n")
    
    # Parse categories
    categories = None
    if args.categories:
        categories = [BrandCategory(cat) for cat in args.categories]
        if not args.quiet:
            print(f"[*] Categories: {', '.join(args.categories)}")
    
    # Parse realism level
    realism = RealismLevel(args.realism)
    if not args.quiet:
        print(f"[*] Realism Level: {args.realism}")
        print(f"[*] Total URLs: {args.count}")
    
    # Generate synthetic URLs
    if not args.quiet:
        print(f"\n{'='*70}")
        print("STEP 1: GENERATING SYNTHETIC URLs")
        print('='*70)
    
    generator = SyntheticURLGenerator(seed=args.seed)
    urls, labels = generator.generate_balanced_dataset(
        total_count=args.count,
        categories=categories,
        realism=realism
    )
    
    if not args.quiet:
        print(f"\n[OK] Generated {len(urls)} URLs")
        print(f"    Phishing URLs: {sum(labels)}")
        print(f"    Legitimate URLs: {len(labels) - sum(labels)}")
    
    # Show sample URLs if requested
    if args.show_samples > 0:
        print(f"\n{'='*70}")
        print(f"SAMPLE URLs (showing {min(args.show_samples, len(urls))})")
        print('='*70)
        
        phishing_shown = 0
        legit_shown = 0
        
        print("\nPhishing URLs:")
        for url, label in zip(urls, labels):
            if label == 1 and phishing_shown < args.show_samples // 2:
                print(f"  {phishing_shown + 1}. {url}")
                phishing_shown += 1
        
        print("\nLegitimate URLs:")
        for url, label in zip(urls, labels):
            if label == 0 and legit_shown < args.show_samples // 2:
                print(f"  {legit_shown + 1}. {url}")
                legit_shown += 1
    
    # Evaluate model
    if not args.quiet:
        print(f"\n{'='*70}")
        print("STEP 2: EVALUATING MODEL PERFORMANCE")
        print('='*70)
    
    evaluator = SyntheticURLEvaluator(model_path=args.model_path)
    
    try:
        results = evaluator.evaluate_urls(urls, labels, verbose=not args.quiet)
        
        # Error analysis
        if args.analyze_errors:
            evaluator.analyze_errors(max_display=args.max_errors)
        
        # Export results
        if args.export:
            # Determine format from extension
            if args.export.endswith('.json'):
                export_format = 'json'
            elif args.export.endswith('.csv'):
                export_format = 'csv'
            else:
                export_format = 'json'
                args.export += '.json'
            
            evaluator.export_results(
                args.export,
                format=export_format,
                include_urls=args.include_urls
            )
        
        # Summary
        if not args.quiet:
            print(f"\n{'='*70}")
            print("TESTING COMPLETED SUCCESSFULLY")
            print('='*70)
            
            metrics = results['metrics']
            print(f"\nQuick Summary:")
            print(f"  ✓ Accuracy:  {metrics['accuracy']:.2%}")
            print(f"  ✓ Precision: {metrics['precision']:.2%}")
            print(f"  ✓ Recall:    {metrics['recall']:.2%}")
            print(f"  ✓ F1-Score:  {metrics['f1_score']:.2%}")
            
            if args.export:
                print(f"\n  📁 Results exported to: {args.export}")
            
            print("\n" + "="*70)
        
        return 0
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure you have trained a model first:")
        print("  python ml/trainer.py")
        print("  or")
        print("  python ml/improved_trainer.py")
        return 1
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
