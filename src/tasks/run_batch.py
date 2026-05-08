
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
    parser.add_argument("--skip-social", action="store_true", help="Skip Social experiments (Dictator, Ultimatum, Trust)")
    parser.add_argument("--skip-cooperation", action="store_true", help="Skip Cooperation experiments (Stag Hunt, Beauty Contest, Centipede, Public Goods, Traveller's Dilemma)")
    
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

        trust_script = os.path.join(base_dir, "trust_game.py")
        run_script(trust_script, ["--model", args.model])

    # 4. Cooperation Experiments
    if not args.skip_cooperation:
        for script_name in ["stag_hunt.py", "beauty_contest.py", "centipede_game.py", "public_goods.py", "travellers_dilemma.py"]:
            run_script(os.path.join(base_dir, script_name), ["--model", args.model])

    # 5. Calculate Rationality Stats (updates _rationality.json for index.html)
    if not args.skip_time and not args.skip_independence:
        print("\n📊 CALCULATING RATIONALITY SCORES...")
        stats_script = os.path.join(project_root, "src", "tools", "calculate_rationality_stats.py")
        run_script(stats_script, [])
    else:
        print("\n⚠️ Skipping rationality score calculation (requires both Time and Independence experiments)")

    print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results recorded for {args.model}")
    print("Run 'cd web && python -m http.server 8000' to view the dashboard.")

if __name__ == "__main__":
    main()
