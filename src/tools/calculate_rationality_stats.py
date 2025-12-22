import json
import os
import numpy as np
from scipy.optimize import curve_fit
import glob

def calculate_time_metrics(model_dir):
    """
    Calculate time preference metrics from discount_rate_results.json
    Returns:
        delta (float): Aggregate discount factor (0-1)
        magnitude_penalty (float): Penalty for magnitude effect (0-5)
    """
    results_path = os.path.join(model_dir, "discount_rate_results.json")
    if not os.path.exists(results_path):
        print(f"  ⚠️ No time results found at {results_path}")
        return None, 0

    with open(results_path, 'r') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            print(f"  ⚠️ Error decoding {results_path}")
            return None, 0

    if not results:
        return None, 0

    # Filter for baseline (front_end_delay = 0)
    baseline = [r for r in results if r.get('front_end_delay', 0) == 0]
    
    if not baseline:
        return None, 0

    # 1. Fit Exponential Model to get aggregate delta
    # Model: discount_factor = delta ^ (delay_days / 365)
    # We have 'implied_discount_factor' in results which is (indifference / larger)
    # But wait, implied_discount_factor in the json is per period? 
    # Let's check the json content again.
    # "discount_factor": 0.0493..., "delay_days": 30.42
    # The JSON's "discount_factor" key seems to be the ratio (indifference/larger).
    # Let's call this 'observed_discount_factor'.
    
    delays = []
    observed_factors = []
    
    amount_groups = {} # For magnitude effect
    
    for r in baseline:
        d = r['delay_days']
        f_obs = r['indifference_amount'] / r['larger_amount'] if r['larger_amount'] > 0 else 1.0
        
        delays.append(d)
        observed_factors.append(f_obs)
        
        amt = r['larger_amount']
        if amt not in amount_groups:
            amount_groups[amt] = []
        amount_groups[amt].append((d, f_obs))

    # Fit exponential: f(t) = delta ^ (t/365)
    def exponential_func(t, delta):
        return delta ** (t / 365.0)

    try:
        popt, _ = curve_fit(exponential_func, delays, observed_factors, p0=[0.9], bounds=(0.01, 1.0))
        delta = popt[0]
    except Exception as e:
        print(f"  ⚠️ Fitting error: {e}")
        delta = np.mean(observed_factors) # Fallback (bad approximation but safe)

    # 2. Calculate Magnitude Effect Penalty
    # Compare raw mean discount factor (or fitted delta) across amounts.
    # If delta gets higher as amount increases, that's the magnitude effect.
    # Score penalty: 5 points for "Strong" effect.
    # EconBench Logic: "If the model changes its discount rate significantly... Penalty: -5"
    
    amounts = sorted(amount_groups.keys())
    if len(amounts) >= 2:
        # Calculate avg annualized rate or just simple avg factor for low vs high
        # To be robust, let's fit delta for the lowest and highest amount groups separately
        
        def fit_delta_for_group(group_data):
            g_delays = [d for d, _ in group_data]
            g_factors = [f for _, f in group_data]
            if len(g_delays) < 2: return 0.5
            try:
                popt, _ = curve_fit(exponential_func, g_delays, g_factors, p0=[0.9], bounds=(0.01, 1.0))
                return popt[0]
            except:
                return np.mean(g_factors)

        delta_low = fit_delta_for_group(amount_groups[amounts[0]])
        delta_high = fit_delta_for_group(amount_groups[amounts[-1]])
        
        # If delta_high is significantly > delta_low, apply penalty
        # Threshold: let's say 0.1 difference
        diff = delta_high - delta_low
        if diff > 0.1:
            magnitude_penalty = 5
        elif diff > 0.05:
            magnitude_penalty = 2.5
        else:
            magnitude_penalty = 0
            
    else:
        magnitude_penalty = 0

    return delta, magnitude_penalty


def calculate_risk_metrics(model_dir):
    """
    Calculate risk consistency metrics from mm_triangle_results.json
    Returns:
        error_rate (float): Percentage deviation from parallel linearity (0-100)
    """
    results_path = os.path.join(model_dir, "mm_triangle_results.json")
    if not os.path.exists(results_path):
        print(f"  ⚠️ No independence results found at {results_path}")
        return 0 # Default to 0 error if missing? Or None?

    with open(results_path, 'r') as f:
        try:
            data = json.load(f)
        except:
            return 0

    if not data:
        return 0

    # "results" list might be inside a dict if I updated the saving, 
    # but the current file on disk (gpt-4o) was a list.
    # The code I read showed it can be a list.
    # However, my plan mentioned creating a new json format. 
    # The file existing on disk is a list.
    
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        return 0

    # Calculate deviation from Expected Utility (EU)
    # The Independence Axiom implies parallel indifference curves.
    # We can calculate the "Error Rate" as the mean deviation of observed indifference points
    # from the EU-predicted points (Expected Value Equality).
    # This matches the "EU Deviation" logic in the independence script.
    
    deviations = []
    
    for r in results:
        ref = r['reference_point']
        indiff_val = r['indifference_value']
        axis = r['axis']
        
        # Calculate EU predicted value
        # EV = p_L * 0 + p_M * 500 + p_H * 1000
        # Reference EV is given.
        # Axis EV match:
        # If Y-axis: p_L=0, p_M=1-p_H, p_H=p_H. EV = (1-p_H)*500 + p_H*1000 = 500 + 500*p_H
        # => p_H = (EV - 500) / 500
        # If X-axis: p_H=0, p_M=1-p_L, p_L=p_L. EV = (1-p_L)*500 + p_L*0 = 500 - 500*p_L
        # => p_L = (500 - EV) / 500
        
        ev = ref['expected_value']
        
        if axis == "Y":
            predicted = (ev - 500) / 500.0
        else:
            predicted = (500 - ev) / 500.0
            
        predicted = max(0.0, min(1.0, predicted))
        
        dev = abs(indiff_val - predicted)
        deviations.append(dev)
        
    if not deviations:
        return 0
        
    mean_deviation = np.mean(deviations)
    
    # Convert to "Error Rate" percentage
    # Mean deviation is in probability space [0,1].
    # Let's map it to 0-100 scale.
    error_rate = mean_deviation * 100
    
    return error_rate


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(base_dir, "data", "results")
    web_data_dir = os.path.join(base_dir, "web", "public", "data")
    
    os.makedirs(web_data_dir, exist_ok=True)
    
    # Find all models in time results (assuming specific structure data/results/time/{model_id})
    # model_id could be a directory name
    time_dir = os.path.join(results_dir, "time")
    model_dirs = glob.glob(os.path.join(time_dir, "*"))
    
    print(f"Found {len(model_dirs)} model directories in {time_dir}")
    
    for m_dir in model_dirs:
        if not os.path.isdir(m_dir): continue
        
        model_id = os.path.basename(m_dir)
        print(f"\nProcessing {model_id}...")
        
        # 1. Time Metrics
        delta, mag_penalty = calculate_time_metrics(m_dir)
        
        # 2. Risk Metrics
        # Need to find corresponding independence directory
        indep_dir = os.path.join(results_dir, "independence", model_id)
        error_rate = calculate_risk_metrics(indep_dir)
        
        print(f"  δ: {delta:.3f}")
        print(f"  Risk Error: {error_rate:.1f}%")
        print(f"  Mag Penalty: {mag_penalty}")
        
        # 3. Save to web JSON
        output_data = {
            "model": model_id,
            "metrics": {
                "patience": {
                    "discount_factor": delta, 
                    "formatted_delta": f"{delta:.2f}"
                },
                "risk": {
                    "error_rate": error_rate,
                    "formatted_error": f"{error_rate:.0f}%"
                },
                "penalties": {
                    "magnitude_effect": mag_penalty
                }
            }
        }
        
        out_path = os.path.join(web_data_dir, f"{model_id}_rationality.json")
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  ✓ Saved {out_path}")

if __name__ == "__main__":
    main()
