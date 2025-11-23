#!/usr/bin/env python3
"""
Scheduled Retraining Script for Adaptive Malware Detection
This script can be run periodically (e.g., daily/weekly) to check if retraining is needed
and automatically retrain models with new data.

Usage:
    python scheduled_retrain.py [--model rf|lstm|both] [--force] [--dry-run]
    
    --model: Which model(s) to retrain (default: both)
    --force: Force retraining even if thresholds not met
    --dry-run: Check if retraining is needed without actually retraining
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.adaptive_learning import AdaptiveLearningManager
from server.performance_monitor import PerformanceMonitor
from Model.adaptive_retrain import AdaptiveRetrainer


def main():
    """Main entry point for scheduled retraining."""
    parser = argparse.ArgumentParser(
        description="Scheduled adaptive retraining for malware detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if retraining is needed (dry run)
  python scheduled_retrain.py --dry-run
  
  # Retrain both models if thresholds met
  python scheduled_retrain.py
  
  # Force retrain Random Forest only
  python scheduled_retrain.py --model rf --force
  
  # Retrain LSTM only if thresholds met
  python scheduled_retrain.py --model lstm
        """
    )
    
    parser.add_argument(
        "--model",
        choices=["rf", "lstm", "both"],
        default="both",
        help="Which model(s) to retrain (default: both)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if thresholds not met"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check if retraining is needed without actually retraining"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("SCHEDULED ADAPTIVE RETRAINING")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model(s): {args.model}")
    print(f"Force: {args.force}")
    print(f"Dry run: {args.dry_run}")
    print("="*70)
    
    try:
        # Initialize components
        adaptive_manager = AdaptiveLearningManager()
        performance_monitor = PerformanceMonitor()
        
        # Get statistics
        stats = adaptive_manager.get_statistics()
        
        print(f"\n📊 Current Statistics:")
        print(f"   New samples (total): {stats['new_samples_total']}")
        print(f"   New samples (with labels): {stats['new_samples_with_labels']}")
        print(f"   New samples (used): {stats['new_samples_used']}")
        print(f"   Feedback (unprocessed): {stats['feedback_unprocessed']}")
        print(f"   Retraining attempts: {stats['retraining_attempts']}")
        
        if stats['last_retraining']:
            print(f"   Last retraining: {stats['last_retraining']['timestamp']}")
            print(f"     Model: {stats['last_retraining']['model_type']}")
            print(f"     Success: {stats['last_retraining']['success']}")
        
        # Check if retraining should be triggered
        should_retrain, reason = adaptive_manager.should_trigger_retraining()
        
        print(f"\n🔍 Retraining Check:")
        print(f"   Should retrain: {should_retrain}")
        print(f"   Reason: {reason}")
        
        if args.dry_run:
            print(f"\n✅ Dry run complete. Retraining {'would be' if (should_retrain or args.force) else 'would NOT be'} triggered.")
            return 0
        
        # Check performance drift
        if args.verbose:
            print(f"\n📈 Checking performance drift...")
            baseline_acc = 0.90
            drift_results = performance_monitor.check_all_drift_indicators(baseline_accuracy=baseline_acc)
            
            if drift_results.get("drift_detected"):
                print(f"   ⚠️ Concept drift detected!")
                print(f"   Details: {drift_results}")
            else:
                print(f"   ✅ No significant drift detected")
        
        # Decide whether to retrain
        if not (should_retrain or args.force):
            print(f"\n⏭️ Skipping retraining:")
            print(f"   - Thresholds not met")
            print(f"   - Use --force to retrain anyway")
            return 0
        
        # Perform retraining
        print(f"\n🚀 Starting retraining...")
        retrainer = AdaptiveRetrainer()
        
        if args.model == "rf":
            print(f"\n📌 Retraining Random Forest model...")
            success, results = retrainer.retrain_random_forest()
            
            if success:
                print(f"\n✅ Random Forest retraining completed successfully!")
                print(f"   Accuracy: {results.get('accuracy', 'N/A'):.4f}")
                print(f"   Improvement: {results.get('improvement', 0)*100:+.2f}%")
            else:
                print(f"\n❌ Random Forest retraining failed")
                print(f"   Error: {results.get('error', 'Unknown error')}")
                return 1
                
        elif args.model == "lstm":
            print(f"\n📌 Retraining LSTM model...")
            success, results = retrainer.retrain_lstm()
            
            if success:
                print(f"\n✅ LSTM retraining completed successfully!")
                print(f"   Accuracy: {results.get('accuracy', 'N/A'):.4f}")
                print(f"   Improvement: {results.get('improvement', 0)*100:+.2f}%")
            else:
                print(f"\n❌ LSTM retraining failed")
                print(f"   Error: {results.get('error', 'Unknown error')}")
                return 1
                
        else:  # both
            print(f"\n📌 Retraining both models...")
            results = retrainer.retrain_both()
            
            rf_success = results.get("rf", {}).get("accuracy") is not None
            lstm_success = results.get("lstm", {}).get("accuracy") is not None
            
            if rf_success:
                print(f"\n✅ Random Forest: Success")
                print(f"   Accuracy: {results['rf'].get('accuracy', 'N/A'):.4f}")
            else:
                print(f"\n❌ Random Forest: Failed")
                print(f"   Error: {results['rf'].get('error', 'Unknown error')}")
            
            if lstm_success:
                print(f"\n✅ LSTM: Success")
                print(f"   Accuracy: {results['lstm'].get('accuracy', 'N/A'):.4f}")
            else:
                print(f"\n❌ LSTM: Failed")
                print(f"   Error: {results['lstm'].get('error', 'Unknown error')}")
            
            if not (rf_success and lstm_success):
                return 1
        
        print(f"\n{'='*70}")
        print(f"✅ SCHEDULED RETRAINING COMPLETE")
        print(f"{'='*70}")
        
        # Get updated statistics
        updated_stats = adaptive_manager.get_statistics()
        print(f"\n📊 Updated Statistics:")
        print(f"   New samples (with labels): {updated_stats['new_samples_with_labels']}")
        print(f"   New samples (used): {updated_stats['new_samples_used']}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during scheduled retraining: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

