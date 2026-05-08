
import argparse
import subprocess
import sys
import os

def run_script(script_path, args):
    """Run a python script with arguments"""
    cmd = [sys.executable, script_path] + args
    print(f"\n🚀 RUNNING: {' '.join(cmd)}")
    print("-" * 70)
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ COMPLETED: {os.path.basename(script_path)}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {os.path.basename(script_path)} failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    print("-" * 70)

def main():
    parser = argparse.ArgumentParser(description="Run all EconBench experiments for a given model")
    parser.add_argument("--model", type=str, required=True, help="Model ID (e.g., gpt-4o, gemini-2.0-flash)")
    parser.add_argument("--skip-independence", action="store_true", help="Skip Independence experiment")
    parser.add_argument("--skip-time", action="store_true", help="Skip Time experiment")
    parser.add_argument("--skip-social", action="store_true", help="Skip Social experiment")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    
    # 1. Independence Experiment
    if not args.skip_independence:
        independence_script = os.path.join(base_dir, "independence.py")
        run_script(independence_script, ["--model", args.model])
        
    # 2. Time Experiment
    if not args.skip_time:
        time_script = os.path.join(base_dir, "time.py")
        run_script(time_script, ["--model", args.model])
        
    # 3. Social Experiments
    if not args.skip_social:
        dictator_script = os.path.join(base_dir, "dictator.py")
        run_script(dictator_script, ["--model", args.model])

        ultimatum_script = os.path.join(base_dir, "ultimatum.py")
        run_script(ultimatum_script, ["--model", args.model])
        
    # 4. Calculate Rationality Stats (updates _rationality.json for index.html)
    if not args.skip_time and not args.skip_independence:
        print("\n📊 CALCULATING RATIONALITY SCORES...")
        stats_script = os.path.join(project_root, "src", "tools", "calculate_rationality_stats.py")
        run_script(stats_script, [])
    else:
        print("\n⚠️ Skipping rationality score calculation (requires both Time and Independence experiments)")

    print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results recorded for {args.model}")

if __name__ == "__main__":
    main()
