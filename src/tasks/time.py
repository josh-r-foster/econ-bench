"""
Discount Rate Elicitation Experiment: Testing Dynamic Consistency in LLM Time Preferences

This script implements a comprehensive elicitation of an LLM's discount rate and tests
for dynamic (in)consistency to distinguish between exponential and hyperbolic discounting.

Key Features:
- Phase 1: Baseline discount rate elicitation via bisection
- Phase 2: Dynamic consistency tests across multiple front-end delays
- Phase 3: Model fitting (exponential, hyperbolic, quasi-hyperbolic)
- Phase 4: Analysis and visualization following MM Triangle style
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()
import sys
import os

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import the model registry factory
from src.models.registry import get_model_interface

# -------------------------------------------------------------
# 1. Configuration & Model Loading
# -------------------------------------------------------------

# Global variables to be initialized in main
model_id = None
llm = None
PRINT_INTERACTIONS = True


# -------------------------------------------------------------
# 2. Experiment Parameters
# -------------------------------------------------------------

# Variety of monetary amounts (base amounts for the "later" option)
AMOUNTS = [10, 100, 1000]  # Different stake sizes


# Time delays in days
DELAYS = [30.42 * 1, 30.42 * 3, 30.42 * 6, 30.42 * 12, 30.42 * 24, 30.42 * 36, 30.42 * 48, 30.42 * 60]

# Front-end delays for dynamic consistency tests
FRONT_END_DELAYS = [0, 30.42 * 1, 30.42 * 12]

# Bisection parameters
N_ITERATIONS = 10  # Gives ~0.1% precision

# -------------------------------------------------------------
# 3. Data Structures
# -------------------------------------------------------------

@dataclass
class IntertemporalChoice:
    """Record of a single binary choice between two time-money pairs"""
    iteration: int
    amount_sooner: float
    amount_later: float
    delay_sooner: int  # Days until sooner payment
    delay_later: int   # Days until later payment
    choice: str  # "SS" (smaller-sooner) or "LL" (larger-later)
    raw_response: str
    lower_bound: float
    upper_bound: float


@dataclass
class DiscountRateResult:
    """Result of one discount rate elicitation"""
    larger_amount: float       # Fixed larger amount
    delay_days: int            # Days between sooner and later
    front_end_delay: int       # Days until sooner option (0 = today)
    indifference_amount: float # Smaller amount that makes agent indifferent
    implied_discount_factor: float  # δ such that x = δ * larger_amount
    implied_annual_rate: float # Annualized discount rate
    n_iterations: int
    choice_history: List[IntertemporalChoice]
    final_precision: float
    swap_order: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def delay_ratio(self) -> float:
        """Ratio of indifference amount to larger amount"""
        return self.indifference_amount / self.larger_amount if self.larger_amount > 0 else 0

    @property
    def delay_months(self) -> float:
        """Delay converted to months (approx 30.42 days/month)"""
        return round(self.delay_days / 30.42, 1)


@dataclass
class DynamicConsistencyTest:
    """Test whether preferences are dynamically consistent"""
    delay_days: int
    larger_amount: float
    immediate_result: DiscountRateResult   # front_end_delay = 0
    delayed_result: DiscountRateResult     # front_end_delay > 0
    front_end_delay: int
    immediate_indiff: float
    delayed_indiff: float
    difference: float  # immediate - delayed (positive = present bias)
    present_bias_detected: bool
    

@dataclass
class MonotonicityCheck:
    """Result of a dominance/monotonicity test"""
    sooner_amount: float
    later_amount: float
    delay_days: int
    expected_choice: str  # What a rational agent should choose
    actual_choice: str
    passed: bool
    raw_response: str





# -------------------------------------------------------------
# 4. Model Interface Adapters
# -------------------------------------------------------------

def generate_response(prompt: str, max_new_tokens: int = 64, temperature: float = 0.01,
                     print_interaction: bool = False) -> str:
    """Generate response from model using the registry interface"""
    # The registry interface returns (response_str, logprobs_dict)
    # We only need the response string here
    response, _ = llm.generate_response(
        prompt, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature,
        verbose=print_interaction
    )
    return response


def parse_ab_choice(response: str) -> Optional[str]:
    """Parse A or B choice using the registry interface's logic"""
    return llm.parse_ab_choice(response)


# -------------------------------------------------------------
# 5. Prompt Templates
# -------------------------------------------------------------

class DiscountRatePrompts:
    """Prompts for the discount rate experiment"""
    
    @staticmethod
    def _format_delay(days: float) -> str:
        """Format delay in days to human readable string"""
        if days <= 0.1:
            return "today"
        if abs(days - 1) < 0.1:
            return "tomorrow"
            
        # Check for years (approx 30.42 * 12 = 365.04)
        years = days / 365.04
        if years >= 0.98 and abs(years - round(years)) < 0.05:
            y = int(round(years))
            return f"in {y} year{'s' if y > 1 else ''}"
            
        # Check for months
        months = days / 30.42
        if months >= 0.98 and abs(months - round(months)) < 0.05:
            m = int(round(months))
            return f"in {m} month{'s' if m > 1 else ''}"
        
        # Check for weeks
        weeks = days / 7
        if weeks >= 0.98 and abs(weeks - round(weeks)) < 0.05:
            w = int(round(weeks))
            return f"in {w} week{'s' if w > 1 else ''}"
            
        # Fallback to days
        return f"in {int(round(days))} days"

    @staticmethod
    def binary_choice(amount_sooner: float, amount_later: float,
                     delay_sooner: float, delay_later: float,
                     swap_order: bool = False) -> str:
        """Create binary choice prompt between two payment options"""
        
        time_sooner = DiscountRatePrompts._format_delay(delay_sooner)
        time_later = DiscountRatePrompts._format_delay(delay_later)
        
        sooner_desc = f"${amount_sooner:.2f} {time_sooner}"
        later_desc = f"${amount_later:.2f} {time_later}"
        
        if swap_order:
            option_a = later_desc
            option_b = sooner_desc
        else:
            option_a = sooner_desc
            option_b = later_desc
        
        return f"""You must choose between two payment options. Which do you prefer?

Option A: {option_a}
Option B: {option_b}

Respond with only the letter "A" or "B".

Answer:"""


# -------------------------------------------------------------
# 6. Bisection Algorithm
# -------------------------------------------------------------

class BisectionElicitor:
    """Implements bisection for finding indifference amounts"""
    
    def __init__(self, n_iterations: int = 10, print_progress: bool = False,
                 print_interactions: bool = True, random_order: bool = True):
        self.n_iterations = n_iterations
        self.print_progress = print_progress
        self.print_interactions = print_interactions
        self.random_order = random_order
    
    def find_indifference_amount(self, larger_amount: float, delay_days: int,
                                  front_end_delay: int = 0,
                                  swap_order: bool = False) -> DiscountRateResult:
        """
        Find the smaller amount that makes agent indifferent to larger_amount after delay.
        
        Args:
            larger_amount: The fixed larger/later amount
            delay_days: Days between sooner and later payment
            front_end_delay: Days until the sooner payment (0 = today)
            swap_order: If True, present later option as A (for bias detection)
        """
        # Bounds: sooner amount ranges from 0 to larger_amount
        lower = 0.0
        upper = larger_amount
        choice_history = []
        
        delay_sooner = front_end_delay
        delay_later = front_end_delay + delay_days
        
        for iteration in range(self.n_iterations):
            midpoint = (lower + upper) / 2
            
            # Randomize order on each iteration if enabled (unless swap_order overrides)
            if self.random_order and not swap_order:
                iter_swap = np.random.random() > 0.5
            else:
                iter_swap = swap_order
            
            prompt = DiscountRatePrompts.binary_choice(
                amount_sooner=midpoint,
                amount_later=larger_amount,
                delay_sooner=delay_sooner,
                delay_later=delay_later,
                swap_order=iter_swap
            )
            
            response = generate_response(prompt, print_interaction=self.print_interactions)
            raw_choice = parse_ab_choice(response)
            
            if self.print_interactions:
                print(f"📋 PARSED CHOICE: {raw_choice if raw_choice else 'FAILED TO PARSE'}")
                if iter_swap:
                    print(f"   (Order swapped: A=later, B=sooner)")
                print("─"*70 + "\n")
            
            if raw_choice is None:
                if self.print_progress:
                    print(f"  ⚠️ Could not parse choice from: {response[:50]}...")
                raw_choice = "A"
            
            # Interpret choice based on iter_swap
            if iter_swap:
                chose_sooner = (raw_choice == "B")
            else:
                chose_sooner = (raw_choice == "A")
            
            semantic_choice = "SS" if chose_sooner else "LL"
            
            choice_record = IntertemporalChoice(
                iteration=iteration + 1,
                amount_sooner=midpoint,
                amount_later=larger_amount,
                delay_sooner=delay_sooner,
                delay_later=delay_later,
                choice=semantic_choice,
                raw_response=response[:200],
                lower_bound=lower,
                upper_bound=upper
            )
            choice_history.append(choice_record)
            
            # Update bounds
            if chose_sooner:
                # Agent prefers sooner, so indifference point is lower
                upper = midpoint
            else:
                # Agent prefers later, so indifference point is higher
                lower = midpoint
            
            if self.print_progress:
                print(f"  Iteration {iteration+1}: midpoint=${midpoint:.2f}, "
                      f"choice={semantic_choice}, bounds=[${lower:.2f}, ${upper:.2f}]")
        
        indifference_amount = (lower + upper) / 2
        
        # Calculate implied discount factor (per delay period)
        discount_factor = indifference_amount / larger_amount if larger_amount > 0 else 1.0
        
        # Annualize: if δ^(delay/365) = discount_factor, then δ = discount_factor^(365/delay)
        if discount_factor > 0 and delay_days > 0:
            annual_factor = discount_factor ** (365 / delay_days)
            annual_rate = (1 / annual_factor) - 1 if annual_factor > 0 else float('inf')
        else:
            annual_rate = 0.0
        
        return DiscountRateResult(
            larger_amount=larger_amount,
            delay_days=delay_days,
            front_end_delay=front_end_delay,
            indifference_amount=indifference_amount,
            implied_discount_factor=discount_factor,
            implied_annual_rate=annual_rate,
            n_iterations=self.n_iterations,
            choice_history=choice_history,
            final_precision=upper - lower,
            swap_order=swap_order
        )


# -------------------------------------------------------------
# 7. Main Experiment Runner
# -------------------------------------------------------------

class DiscountRateExperiment:
    """Main experiment class for discount rate elicitation"""
    
    def __init__(self, amounts: List[float] = None, delays: List[int] = None,
                 front_end_delays: List[int] = None, n_iterations: int = 10,
                 validation_fraction: float = 0.1, print_progress: bool = True,
                 run_diagnostics: bool = True):
        """
        Args:
            amounts: List of larger amounts to test
            delays: List of delays in days
            front_end_delays: List of front-end delays for consistency tests
            n_iterations: Bisection iterations
            validation_fraction: Fraction of points to re-test
            print_progress: Whether to print updates
            run_diagnostics: Whether to run consistency/monotonicity checks
        """
        self.amounts = amounts or AMOUNTS
        self.delays = delays or DELAYS
        self.front_end_delays = front_end_delays or FRONT_END_DELAYS
        self.n_iterations = n_iterations
        self.validation_fraction = validation_fraction
        self.print_progress = print_progress
        self.run_diagnostics = run_diagnostics
        
        self.elicitor = BisectionElicitor(
            n_iterations=n_iterations,
            print_progress=print_progress,
            print_interactions=PRINT_INTERACTIONS
        )
        
        self.results: List[DiscountRateResult] = []
        self.validation_results: List[Tuple[DiscountRateResult, DiscountRateResult]] = []
        self.consistency_tests: List[DynamicConsistencyTest] = []
        self.monotonicity_checks: List[MonotonicityCheck] = []

        self.bidirectional_results: List[Tuple[DiscountRateResult, DiscountRateResult]] = []
    

    
    def run_baseline_elicitation(self) -> List[DiscountRateResult]:
        """Phase 1: Elicit discount rates for all amount/delay combinations"""
        print("\nPHASE 1: Baseline Discount Rate Elicitation")
        print("-"*70)
        
        # Only use front_end_delay = 0 for baseline
        total_points = len(self.amounts) * len(self.delays)
        print(f"  Amounts: {self.amounts}")
        print(f"  Delays: {self.delays}")
        print(f"  Total elicitations: {total_points}")
        print()
        
        for amount in tqdm(self.amounts, desc="Amounts"):
            for delay in tqdm(self.delays, desc="Delays", leave=False):
                result = self.elicitor.find_indifference_amount(
                    larger_amount=amount,
                    delay_days=delay,
                    front_end_delay=0
                )
                self.results.append(result)
                
                if self.print_progress:
                    tqdm.write(f"  ${amount} in {delay}d → indiff ${result.indifference_amount:.2f} "
                              f"(δ={result.implied_discount_factor:.4f})")
        
        return self.results
    
    def run_dynamic_consistency_tests(self) -> List[DynamicConsistencyTest]:
        """Phase 2: Test for dynamic consistency across front-end delays"""
        print("\nPHASE 2: Dynamic Consistency Tests")
        print("-"*70)
        
        # For each amount/delay with front_end_delay=0, compare with delayed versions
        non_zero_fed = [fed for fed in self.front_end_delays if fed > 0]
        print(f"  Front-end delays to test: {non_zero_fed}")
        print()
        
        baseline_results = {(r.larger_amount, r.delay_days): r 
                          for r in self.results if r.front_end_delay == 0}
        
        for amount in tqdm(self.amounts, desc="Consistency tests"):
            for delay in self.delays:
                baseline = baseline_results.get((amount, delay))
                if baseline is None:
                    continue
                
                for fed in non_zero_fed:
                    # Elicit with front-end delay
                    delayed_result = self.elicitor.find_indifference_amount(
                        larger_amount=amount,
                        delay_days=delay,
                        front_end_delay=fed
                    )
                    
                    diff = baseline.indifference_amount - delayed_result.indifference_amount
                    # Positive diff = more impatient for immediate (present bias)
                    present_bias = diff > (amount * 0.02)  # 2% threshold
                    
                    test = DynamicConsistencyTest(
                        delay_days=delay,
                        larger_amount=amount,
                        immediate_result=baseline,
                        delayed_result=delayed_result,
                        front_end_delay=fed,
                        immediate_indiff=baseline.indifference_amount,
                        delayed_indiff=delayed_result.indifference_amount,
                        difference=diff,
                        present_bias_detected=present_bias
                    )
                    self.consistency_tests.append(test)
                    
                    if self.print_progress:
                        status = "⚠️ BIAS" if present_bias else "✓ OK"
                        tqdm.write(f"  ${amount}/{delay}d: immed=${baseline.indifference_amount:.2f}, "
                                  f"fed{fed}=${delayed_result.indifference_amount:.2f} {status}")
        
        n_biased = sum(1 for t in self.consistency_tests if t.present_bias_detected)
        print(f"\n  Present bias detected in {n_biased}/{len(self.consistency_tests)} tests")
        
        return self.consistency_tests
    
    def run_monotonicity_checks(self, n_checks: int = 5) -> List[MonotonicityCheck]:
        """Run dominance sanity checks"""
        print("\nMONOTONICITY CHECKS: Dominance Tests")
        print("-"*70)
        
        # Test pairs where one option clearly dominates
        test_cases = [
            # (sooner_amt, later_amt, delay) - later should be chosen if later > sooner
            (50, 100, 7),   # $50 today vs $100 in 1 week
            (80, 100, 30),  # $80 today vs $100 in 1 month
            (95, 100, 7),   # Close call: $95 today vs $100 in 1 week
            (100, 100, 30), # Same amount: should prefer today
            (110, 100, 30), # More today: should prefer today
        ][:n_checks]
        
        for sooner, later, delay in tqdm(test_cases, desc="Monotonicity"):
            prompt = DiscountRatePrompts.binary_choice(
                amount_sooner=sooner,
                amount_later=later,
                delay_sooner=0,
                delay_later=delay
            )
            response = generate_response(prompt, print_interaction=PRINT_INTERACTIONS)
            choice = parse_ab_choice(response)
            
            # Determine expected choice
            if sooner >= later:
                expected = "A"  # Sooner (same or more money now)
            else:
                # Later has more money - reasonable to prefer either
                expected = "EITHER"
            
            if expected == "EITHER":
                passed = True
            else:
                passed = (choice == expected)
            
            check = MonotonicityCheck(
                sooner_amount=sooner,
                later_amount=later,
                delay_days=delay,
                expected_choice=expected,
                actual_choice=choice or "FAILED",
                passed=passed,
                raw_response=response[:200]
            )
            self.monotonicity_checks.append(check)
            
            status = "✓" if passed else "✗"
            if self.print_progress:
                tqdm.write(f"  {status}: ${sooner} now vs ${later} in {delay}d → {choice}")
        
        n_passed = sum(1 for c in self.monotonicity_checks if c.passed)
        print(f"\n  Passed: {n_passed}/{len(self.monotonicity_checks)}")
        
        return self.monotonicity_checks
    
    def run_bidirectional_tests(self, n_samples: int = 5) -> List[Tuple[DiscountRateResult, DiscountRateResult]]:
        """Test for positional bias by re-running with swapped order"""
        print("\nBIDIRECTIONAL TESTS: Positional Bias Detection")
        print("-"*70)
        
        if len(self.results) == 0:
            print("  No baseline results to test")
            return []
        
        sample_indices = np.random.choice(len(self.results), min(n_samples, len(self.results)), replace=False)
        
        for idx in tqdm(sample_indices, desc="Bidirectional"):
            original = self.results[idx]
            
            swapped = self.elicitor.find_indifference_amount(
                larger_amount=original.larger_amount,
                delay_days=original.delay_days,
                front_end_delay=original.front_end_delay,
                swap_order=True
            )
            
            self.bidirectional_results.append((original, swapped))
            
            diff = abs(original.indifference_amount - swapped.indifference_amount)
            if self.print_progress:
                tqdm.write(f"  ${original.larger_amount}/{original.delay_days}d: "
                          f"normal=${original.indifference_amount:.2f}, "
                          f"swapped=${swapped.indifference_amount:.2f}, diff=${diff:.2f}")
        
        if self.bidirectional_results:
            diffs = [abs(o.indifference_amount - s.indifference_amount) 
                    for o, s in self.bidirectional_results]
            mean_diff = np.mean(diffs)
            print(f"\n  Mean difference: ${mean_diff:.2f}")
            print(f"  Positional bias: {'DETECTED' if mean_diff > 5 else 'MINIMAL'}")
        
        return self.bidirectional_results
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment"""
        print("\n" + "="*70)
        print(" "*10 + "DISCOUNT RATE ELICITATION EXPERIMENT")
        print(" "*15 + f"Model: {model_id}")
        print("="*70 + "\n")
        
        # Phase 1: Baseline
        self.run_baseline_elicitation()
        
        # Phase 2: Dynamic Consistency Tests
        if self.run_diagnostics:
            self.run_dynamic_consistency_tests()
            self.run_monotonicity_checks(n_checks=5)
            self.run_bidirectional_tests(n_samples=5)
        
        # Validation: Re-test subset
        print("\nVALIDATION: Consistency Check")
        print("-"*70)
        n_validation = max(1, int(len(self.results) * self.validation_fraction))
        validation_indices = np.random.choice(len(self.results), n_validation, replace=False)
        
        for idx in tqdm(validation_indices, desc="Validation"):
            original = self.results[idx]
            retest = self.elicitor.find_indifference_amount(
                larger_amount=original.larger_amount,
                delay_days=original.delay_days,
                front_end_delay=original.front_end_delay
            )
            self.validation_results.append((original, retest))
            
            dev = abs(original.indifference_amount - retest.indifference_amount)
            if self.print_progress:
                tqdm.write(f"  ${original.larger_amount}/{original.delay_days}d: "
                          f"orig=${original.indifference_amount:.2f}, "
                          f"retest=${retest.indifference_amount:.2f}, dev=${dev:.2f}")
        
        deviations = [abs(o.indifference_amount - r.indifference_amount) 
                     for o, r in self.validation_results]
        mean_dev = np.mean(deviations) if deviations else 0
        print(f"\n  Mean deviation: ${mean_dev:.2f}")
        
        # Phase 3: Analysis
        print("\nPHASE 3: Analysis")
        print("-"*70)
        analysis = self.analyze_results()
        
        return analysis
    
    def analyze_results(self) -> Dict[str, Any]:
        """Perform analysis on collected data"""
        
        # Fit discount models
        model_fits = self._fit_discount_models()
        
        # Calculate summary statistics
        baseline_results = [r for r in self.results if r.front_end_delay == 0]
        
        analysis = {
            "n_baseline_elicitations": len(baseline_results),
            "n_consistency_tests": len(self.consistency_tests),
            "amounts_tested": list(set(r.larger_amount for r in self.results)),
            "delays_tested": list(set(r.delay_days for r in self.results)),
            "model_fits": model_fits,
            "validation": {
                "n_retests": len(self.validation_results),
                "mean_deviation": float(np.mean([abs(o.indifference_amount - r.indifference_amount) 
                                                  for o, r in self.validation_results])) if self.validation_results else 0.0,
            },
            "diagnostics": {
                "monotonicity": {
                    "n_checks": len(self.monotonicity_checks),
                    "n_passed": sum(1 for c in self.monotonicity_checks if c.passed),
                    "pass_rate": sum(1 for c in self.monotonicity_checks if c.passed) / len(self.monotonicity_checks) if self.monotonicity_checks else 1.0
                },
                "present_bias": {
                    "n_tests": len(self.consistency_tests),
                    "n_biased": sum(1 for t in self.consistency_tests if t.present_bias_detected),
                    "bias_rate": sum(1 for t in self.consistency_tests if t.present_bias_detected) / len(self.consistency_tests) if self.consistency_tests else 0.0
                },
                "bidirectional": {
                    "n_samples": len(self.bidirectional_results),
                    "mean_difference": float(np.mean([abs(o.indifference_amount - s.indifference_amount) 
                                                       for o, s in self.bidirectional_results])) if self.bidirectional_results else 0.0
                }
            }
        }
        
        # Print summary
        print(f"  Baseline elicitations: {analysis['n_baseline_elicitations']}")
        print(f"  Dynamic consistency tests: {analysis['n_consistency_tests']}")
        print()
        
        print("  MODEL FITS:")
        for model_name, fit in model_fits.items():
            if isinstance(fit, dict) and 'sse' in fit:
                print(f"    {model_name}: SSE={fit['sse']:.6f}")
        print()
        
        print("  DISCOUNTING TYPE CLASSIFICATION:")
        best_model = model_fits.get('best_model', 'unknown')
        print(f"    Best fitting model: {best_model}")
        
        if 'quasi_hyperbolic' in model_fits and 'beta' in model_fits['quasi_hyperbolic']:
            beta = model_fits['quasi_hyperbolic']['beta']
            delta = model_fits['quasi_hyperbolic']['delta']
            print(f"    β (present bias): {beta:.4f}")
            print(f"    δ (exponential factor): {delta:.4f}")
            if beta < 0.95:
                print(f"    → PRESENT BIAS DETECTED (β < 1)")
            else:
                print(f"    → APPROXIMATELY TIME-CONSISTENT")
        print()
        
        diag = analysis['diagnostics']
        print("  DIAGNOSTICS:")
        print(f"    Monotonicity: {diag['monotonicity']['n_passed']}/{diag['monotonicity']['n_checks']} passed")
        print(f"    Present bias detected in: {diag['present_bias']['n_biased']}/{diag['present_bias']['n_tests']} tests")
        print()
        
        return analysis
    
    def _fit_discount_models(self) -> Dict[str, Any]:
        """Fit exponential, hyperbolic, and quasi-hyperbolic models"""
        baseline = [r for r in self.results if r.front_end_delay == 0]
        
        if len(baseline) < 3:
            return {"status": "INSUFFICIENT_DATA"}
        
        # Prepare data: (delay_days, discount_factor)
        delays = np.array([r.delay_days for r in baseline])
        factors = np.array([r.implied_discount_factor for r in baseline])
        
        results = {}
        
        # 1. Exponential: δ^(t/365)
        def exponential(t, delta):
            return delta ** (t / 365)
        
        try:
            popt, _ = curve_fit(exponential, delays, factors, p0=[0.95], bounds=(0.01, 1.0))
            delta_exp = popt[0]
            pred_exp = exponential(delays, delta_exp)
            sse_exp = np.sum((factors - pred_exp) ** 2)
            results['exponential'] = {
                'delta': float(delta_exp),
                'annual_rate': float((1/delta_exp) - 1) if delta_exp > 0 else float('inf'),
                'sse': float(sse_exp),
                'predictions': pred_exp.tolist()
            }
        except Exception as e:
            results['exponential'] = {'error': str(e)}
        
        # 2. Hyperbolic: 1/(1 + k*t)
        def hyperbolic(t, k):
            return 1 / (1 + k * t)
        
        try:
            popt, _ = curve_fit(hyperbolic, delays, factors, p0=[0.01], bounds=(0.0001, 1.0))
            k_hyp = popt[0]
            pred_hyp = hyperbolic(delays, k_hyp)
            sse_hyp = np.sum((factors - pred_hyp) ** 2)
            results['hyperbolic'] = {
                'k': float(k_hyp),
                'sse': float(sse_hyp),
                'predictions': pred_hyp.tolist()
            }
        except Exception as e:
            results['hyperbolic'] = {'error': str(e)}
        
        # 3. Quasi-hyperbolic: β * δ^(t/365) for t > 0
        def quasi_hyperbolic(t, beta, delta):
            return np.where(t > 0, beta * (delta ** (t / 365)), 1.0)
        
        try:
            popt, _ = curve_fit(quasi_hyperbolic, delays, factors, 
                               p0=[0.9, 0.95], bounds=([0.01, 0.01], [1.0, 1.0]))
            beta_qh, delta_qh = popt
            pred_qh = quasi_hyperbolic(delays, beta_qh, delta_qh)
            sse_qh = np.sum((factors - pred_qh) ** 2)
            results['quasi_hyperbolic'] = {
                'beta': float(beta_qh),
                'delta': float(delta_qh),
                'annual_rate': float((1/delta_qh) - 1) if delta_qh > 0 else float('inf'),
                'sse': float(sse_qh),
                'predictions': pred_qh.tolist()
            }
        except Exception as e:
            results['quasi_hyperbolic'] = {'error': str(e)}
        
        # Determine best model (lowest SSE)
        sse_values = {}
        for model, fit in results.items():
            if isinstance(fit, dict) and 'sse' in fit:
                sse_values[model] = fit['sse']
        
        if sse_values:
            results['best_model'] = min(sse_values, key=sse_values.get)
        
        return results
    
    def generate_visualizations(self, output_dir: str = "."):
        """Generate all visualization plots"""
        print("\nGenerating visualizations...")
        print("-"*70)
        
        # 1. Discount curve by delay
        self._plot_discount_curve(os.path.join(output_dir, "discount_curve.png"))
        
        # 2. Model comparison
        self._plot_model_comparison(os.path.join(output_dir, "model_comparison.png"))
        
        # 3. Dynamic consistency heatmap
        self._plot_consistency_heatmap(os.path.join(output_dir, "dynamic_consistency.png"))
        
        # 4. Amount sensitivity
        self._plot_amount_sensitivity(os.path.join(output_dir, "amount_sensitivity.png"))
        
        # 5. Present bias detection
        self._plot_present_bias(os.path.join(output_dir, "present_bias.png"))
        
        print()
    
    def _plot_discount_curve(self, save_path: str):
        """Plot elicited discount factors over time"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        baseline = [r for r in self.results if r.front_end_delay == 0]
        
        # Group by amount
        amounts = sorted(set(r.larger_amount for r in baseline))
        colors = plt.cm.viridis(np.linspace(0, 1, len(amounts)))
        
        for amount, color in zip(amounts, colors):
            amount_results = [r for r in baseline if r.larger_amount == amount]
            delays = [r.delay_days for r in amount_results]
            factors = [r.implied_discount_factor for r in amount_results]
            
            # Sort by delay
            sorted_pairs = sorted(zip(delays, factors))
            delays, factors = zip(*sorted_pairs) if sorted_pairs else ([], [])
            
            ax.plot(delays, factors, 'o-', color=color, label=f'${amount}', 
                   markersize=8, linewidth=2)
        
        ax.set_xlabel('Delay (days)', fontsize=12)
        ax.set_ylabel('Discount Factor (indiff/later)', fontsize=12)
        ax.set_title('Elicited Discount Factors by Delay', fontsize=14, fontweight='bold')
        ax.legend(title='Amount')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _plot_model_comparison(self, save_path: str):
        """Plot fitted model curves against data"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        baseline = [r for r in self.results if r.front_end_delay == 0]
        if not baseline:
            return
        
        # Plot data points
        delays = np.array([r.delay_days for r in baseline])
        factors = np.array([r.implied_discount_factor for r in baseline])
        ax.scatter(delays, factors, c='black', s=100, zorder=5, label='Observed', alpha=0.7)
        
        # Generate smooth curves for models
        t_smooth = np.linspace(1, max(delays) * 1.1, 100)
        
        model_fits = self._fit_discount_models()
        
        # Exponential
        if 'exponential' in model_fits and 'delta' in model_fits['exponential']:
            delta = model_fits['exponential']['delta']
            y_exp = delta ** (t_smooth / 365)
            ax.plot(t_smooth, y_exp, '--', color='blue', linewidth=2, 
                   label=f"Exponential (δ={delta:.3f})")
        
        # Hyperbolic
        if 'hyperbolic' in model_fits and 'k' in model_fits['hyperbolic']:
            k = model_fits['hyperbolic']['k']
            y_hyp = 1 / (1 + k * t_smooth)
            ax.plot(t_smooth, y_hyp, ':', color='red', linewidth=2,
                   label=f"Hyperbolic (k={k:.4f})")
        
        # Quasi-hyperbolic
        if 'quasi_hyperbolic' in model_fits and 'beta' in model_fits['quasi_hyperbolic']:
            beta = model_fits['quasi_hyperbolic']['beta']
            delta = model_fits['quasi_hyperbolic']['delta']
            y_qh = beta * (delta ** (t_smooth / 365))
            ax.plot(t_smooth, y_qh, '-', color='green', linewidth=2,
                   label=f"Quasi-hyperbolic (β={beta:.3f}, δ={delta:.3f})")
        
        ax.set_xlabel('Delay (days)', fontsize=12)
        ax.set_ylabel('Discount Factor', fontsize=12)
        ax.set_title('Discount Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Add best model annotation
        if 'best_model' in model_fits:
            ax.text(0.95, 0.95, f"Best fit: {model_fits['best_model']}", 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _plot_consistency_heatmap(self, save_path: str):
        """Plot heatmap of dynamic consistency test results"""
        if not self.consistency_tests:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create matrix: rows = delays, cols = front-end delays
        delays = sorted(set(t.delay_days for t in self.consistency_tests))
        feds = sorted(set(t.front_end_delay for t in self.consistency_tests))
        
        # Average across amounts for each delay/fed combination
        matrix = np.zeros((len(delays), len(feds)))
        for i, d in enumerate(delays):
            for j, f in enumerate(feds):
                tests = [t for t in self.consistency_tests if t.delay_days == d and t.front_end_delay == f]
                if tests:
                    # Positive = present bias (more impatient for immediate)
                    avg_diff = np.mean([t.difference for t in tests])
                    matrix[i, j] = avg_diff
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        
        ax.set_xticks(range(len(feds)))
        ax.set_xticklabels([f'{f}d' for f in feds])
        ax.set_yticks(range(len(delays)))
        ax.set_yticklabels([f'{d}d' for d in delays])
        
        ax.set_xlabel('Front-End Delay', fontsize=12)
        ax.set_ylabel('Payment Delay', fontsize=12)
        ax.set_title('Dynamic Consistency: Indifference Difference\n(Red = Present Bias)', 
                    fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Indiff(immediate) - Indiff(delayed)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _plot_amount_sensitivity(self, save_path: str):
        """Plot how discount rates vary with stake size"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        baseline = [r for r in self.results if r.front_end_delay == 0]
        
        # Group by delay
        delays = sorted(set(r.delay_days for r in baseline))
        colors = plt.cm.plasma(np.linspace(0, 1, len(delays)))
        
        for delay, color in zip(delays, colors):
            delay_results = [r for r in baseline if r.delay_days == delay]
            amounts = [r.larger_amount for r in delay_results]
            factors = [r.implied_discount_factor for r in delay_results]
            
            sorted_pairs = sorted(zip(amounts, factors))
            if sorted_pairs:
                amounts, factors = zip(*sorted_pairs)
                ax.plot(amounts, factors, 'o-', color=color, label=f'{delay}d delay',
                       markersize=8, linewidth=2)
        
        ax.set_xlabel('Stake Size ($)', fontsize=12)
        ax.set_ylabel('Discount Factor', fontsize=12)
        ax.set_title('Magnitude Effect: Discount Factor by Stake Size', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Delay')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _plot_present_bias(self, save_path: str):
        """Plot comparison of immediate vs delayed baseline choices"""
        if not self.consistency_tests:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Scatter plot comparing immediate vs delayed indifference
        ax = axes[0]
        immediate = [t.immediate_indiff for t in self.consistency_tests]
        delayed = [t.delayed_indiff for t in self.consistency_tests]
        biased = [t.present_bias_detected for t in self.consistency_tests]
        
        colors = ['red' if b else 'green' for b in biased]
        ax.scatter(delayed, immediate, c=colors, s=80, alpha=0.7)
        
        # Add diagonal line (consistency)
        max_val = max(max(immediate), max(delayed)) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Consistent')
        
        ax.set_xlabel('Delayed Baseline Indifference ($)', fontsize=12)
        ax.set_ylabel('Immediate Baseline Indifference ($)', fontsize=12)
        ax.set_title('Present Bias Detection\n(Above line = more impatient for immediate)', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Right: Bar chart of bias rate by front-end delay
        ax = axes[1]
        feds = sorted(set(t.front_end_delay for t in self.consistency_tests))
        bias_rates = []
        for fed in feds:
            tests = [t for t in self.consistency_tests if t.front_end_delay == fed]
            rate = sum(1 for t in tests if t.present_bias_detected) / len(tests) if tests else 0
            bias_rates.append(rate)
        
        bars = ax.bar(range(len(feds)), bias_rates, color='steelblue')
        ax.set_xticks(range(len(feds)))
        ax.set_xticklabels([f'{f}d' for f in feds])
        ax.set_xlabel('Front-End Delay', fontsize=12)
        ax.set_ylabel('Present Bias Detection Rate', fontsize=12)
        ax.set_title('Bias Rate by Front-End Delay', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def save_results(self, output_dir: str = "."):
        """Save all results to files"""
        print("\nSaving results...")
        print("-"*70)
        
        # 1. CSV with main results
        csv_data = []
        for result in self.results:
            csv_data.append({
                'larger_amount': result.larger_amount,
                'delay_days': result.delay_days,
                'front_end_delay': result.front_end_delay,
                'indifference_amount': result.indifference_amount,
                'discount_factor': result.implied_discount_factor,
                'annual_rate': result.implied_annual_rate,
                'n_iterations': result.n_iterations,
                'final_precision': result.final_precision,
                'timestamp': result.timestamp
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, "discount_rate_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_path}")
        
        # 2. Detailed JSON
        json_data = []
        for result in self.results:
            json_data.append({
                'larger_amount': result.larger_amount,
                'delay_days': result.delay_days,
                'front_end_delay': result.front_end_delay,
                'indifference_amount': result.indifference_amount,
                'discount_factor': result.implied_discount_factor,
                'annual_rate': result.implied_annual_rate,
                'choice_history': [
                    {
                        'iteration': c.iteration,
                        'amount_sooner': c.amount_sooner,
                        'choice': c.choice,
                        'response': c.raw_response
                    }
                    for c in result.choice_history
                ]
            })
        
        json_path = os.path.join(output_dir, "discount_rate_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"  ✓ Saved: {json_path}")
        
        # 3. Consistency test results
        if self.consistency_tests:
            consistency_data = []
            for test in self.consistency_tests:
                consistency_data.append({
                    'larger_amount': test.larger_amount,
                    'delay_days': test.delay_days,
                    'front_end_delay': test.front_end_delay,
                    'immediate_indiff': test.immediate_indiff,
                    'delayed_indiff': test.delayed_indiff,
                    'difference': test.difference,
                    'present_bias': test.present_bias_detected
                })
            
            cons_df = pd.DataFrame(consistency_data)
            cons_path = os.path.join(output_dir, "consistency_tests.csv")
            cons_df.to_csv(cons_path, index=False)
            print(f"  ✓ Saved: {cons_path}")
            
        # 4. Chart.js compatible JSON
        # We don't save this here by default anymore, as it needs a specific web directory
        # It is called explicitly from main() with the correct path
        
        print()
        
        print()

    def save_chart_data(self, output_dir: str, analysis: Dict[str, Any] = None):
        """Save results in a format optimized for the requested Chart.js visualization"""
        baseline = [r for r in self.results if r.front_end_delay == 0]
        
        # Prepare datasets structure
        chart_data = {
            "labels": [0, 1, 3, 6, 12, 24, 36, 48, 60],
            "datasets": {}
        }
        
        # Group by amount
        amounts = sorted(list(set(r.larger_amount for r in baseline)))
        
        for amount in amounts:
            # Get points for this amount
            amount_points = {0: 1.0} # Start with 0 delay = 1.0
            for r in baseline:
                if r.larger_amount == amount:
                    month = int(round(r.delay_days / 30.42))
                    amount_points[month] = r.implied_discount_factor
            
            # Fill expected delays
            series = []
            for d in chart_data["labels"]:
                series.append(amount_points.get(d, None))
                
            chart_data["datasets"][f"amount_{int(amount)}"] = series
            
        # Add analysis text if available
        if analysis:
            best_model = analysis.get('model_fits', {}).get('best_model', 'unknown')
            
            # Construct text
            text = f"> DETAILS<br><br>Analysis indicates the model is best described by <b>{best_model.upper()} DISCOUNTING</b>."
            
            if 'quasi_hyperbolic' in analysis.get('model_fits', {}) and 'beta' in analysis['model_fits']['quasi_hyperbolic']:
                beta = analysis['model_fits']['quasi_hyperbolic']['beta']
                delta = analysis['model_fits']['quasi_hyperbolic']['delta']
                if beta < 0.95:
                    text += f" Significant present bias was detected (β={beta:.2f}), suggesting impulsivity for immediate rewards."
                else:
                    text += f" No significant present bias was detected (β={beta:.2f})."
                    
            if best_model == 'exponential':
                text += " This behavior is consistent with standard economic theory (time-consistency)."
            elif best_model == 'hyperbolic':
                k = analysis.get('model_fits', {}).get('hyperbolic', {}).get('k', 0)
                text += f" The estimated discount parameter is k={k:.2f}."
                
            chart_data["analysis_text"] = text
            
            chart_data["datasets"][f"amount_{int(amount)}"] = series
            
        os.makedirs(output_dir, exist_ok=True)
        # Use the global model_id to create a unique filename
        model_name_safe = model_id.replace("/", "_").replace(":", "_")
        chart_path = os.path.join(output_dir, f"time_experiment_{model_name_safe}.json")
        with open(chart_path, 'w') as f:
            json.dump(chart_data, f, indent=2)
        print(f"  ✓ Saved: {chart_path} (optimized for Chart.js)")
        
        # Update models registry (models.json)
        registry_path = os.path.join(output_dir, "models.json")
        models_list = []
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    models_list = json.load(f)
            except Exception:
                models_list = []
        
        if model_id not in models_list:
            models_list.append(model_id)
            with open(registry_path, 'w') as f:
                json.dump(models_list, f, indent=2)
            print(f"  ✓ Updated models registry: {model_id} added")
    
    def generate_report(self, analysis: Dict, output_dir: str = "."):
        """Generate comprehensive text report"""
        report_path = os.path.join(output_dir, "discount_rate_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DISCOUNT RATE ELICITATION EXPERIMENT REPORT\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("EXPERIMENT SETUP\n")
            f.write("-"*70 + "\n")
            f.write(f"Amounts tested: {analysis.get('amounts_tested', [])}\n")
            f.write(f"Delays tested: {analysis.get('delays_tested', [])} days\n")
            f.write(f"Front-end delays: {self.front_end_delays}\n")
            f.write(f"Bisection iterations: {self.n_iterations}\n")
            f.write(f"Baseline elicitations: {analysis.get('n_baseline_elicitations', 0)}\n")
            f.write(f"Dynamic consistency tests: {analysis.get('n_consistency_tests', 0)}\n\n")
            
            f.write("VALIDATION (CONSISTENCY CHECK)\n")
            f.write("-"*70 + "\n")
            val = analysis.get('validation', {})
            f.write(f"Retests performed: {val.get('n_retests', 0)}\n")
            f.write(f"Mean deviation: ${val.get('mean_deviation', 0):.2f}\n\n")
            
            f.write("MODEL FITS\n")
            f.write("-"*70 + "\n")
            model_fits = analysis.get('model_fits', {})
            
            if 'exponential' in model_fits and 'delta' in model_fits['exponential']:
                exp = model_fits['exponential']
                f.write(f"Exponential: δ = {exp['delta']:.4f}, Annual Rate = {exp['annual_rate']:.2%}, SSE = {exp['sse']:.6f}\n")
            
            if 'hyperbolic' in model_fits and 'k' in model_fits['hyperbolic']:
                hyp = model_fits['hyperbolic']
                f.write(f"Hyperbolic: k = {hyp['k']:.6f}, SSE = {hyp['sse']:.6f}\n")
            
            if 'quasi_hyperbolic' in model_fits and 'beta' in model_fits['quasi_hyperbolic']:
                qh = model_fits['quasi_hyperbolic']
                f.write(f"Quasi-hyperbolic: β = {qh['beta']:.4f}, δ = {qh['delta']:.4f}, SSE = {qh['sse']:.6f}\n")
            
            f.write(f"\nBest fitting model: {model_fits.get('best_model', 'unknown')}\n\n")
            
            f.write("DIAGNOSTICS\n")
            f.write("-"*70 + "\n")
            diag = analysis.get('diagnostics', {})
            
            mono = diag.get('monotonicity', {})
            f.write(f"Monotonicity checks: {mono.get('n_passed', 0)}/{mono.get('n_checks', 0)} passed\n")
            
            bias = diag.get('present_bias', {})
            f.write(f"Present bias detected: {bias.get('n_biased', 0)}/{bias.get('n_tests', 0)} tests ({bias.get('bias_rate', 0):.1%})\n")
            
            bidir = diag.get('bidirectional', {})
            f.write(f"Positional bias (mean diff): ${bidir.get('mean_difference', 0):.2f}\n\n")
            
            f.write("OVERALL CONCLUSION\n")
            f.write("-"*70 + "\n")
            
            best_model = model_fits.get('best_model', 'unknown')
            if best_model == 'exponential':
                f.write("✓ The LLM appears to be a TIME-CONSISTENT exponential discounter.\n")
                f.write("  Dynamic consistency holds: preferences don't reverse over time.\n")
            elif best_model == 'hyperbolic':
                f.write("✗ The LLM exhibits HYPERBOLIC DISCOUNTING.\n")
                f.write("  This implies dynamic inconsistency and preference reversals.\n")
            elif best_model == 'quasi_hyperbolic':
                qh = model_fits.get('quasi_hyperbolic', {})
                beta = qh.get('beta', 1.0)
                if beta < 0.95:
                    f.write("✗ The LLM exhibits PRESENT BIAS (quasi-hyperbolic discounting).\n")
                    f.write(f"  β = {beta:.4f} indicates {(1-beta)*100:.1f}% present bias.\n")
                    f.write("  This implies time inconsistency for immediate decisions.\n")
                else:
                    f.write("✓ The LLM is approximately TIME-CONSISTENT.\n")
                    f.write(f"  β = {beta:.4f} is close to 1, indicating minimal present bias.\n")
            else:
                f.write("? Unable to conclusively classify discounting behavior.\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"  ✓ Saved: {report_path}")


# -------------------------------------------------------------
# 8. Main Execution
# -------------------------------------------------------------

def main():
    """Main execution function"""
    global model_id, llm, PRINT_INTERACTIONS

    parser = argparse.ArgumentParser(description="Discount Rate Elicitation Experiment")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model ID to use (e.g., gpt-4o, meta-llama/Llama-3.1-70B-Instruct)")
    parser.add_argument("--no-print", action="store_true", help="Disable printing interactions")
    args = parser.parse_args()

    model_id = args.model
    PRINT_INTERACTIONS = not args.no_print

    print(f"Loading model interface for: {model_id}")
    try:
        llm = get_model_interface(model_id)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure the model is defined in src/models/registry.py or is a valid OpenAI model.")
        sys.exit(1)
    
    # Create output directory: data/results/time/{model_id}
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_name_safe = model_id.replace("/", "_").replace(":", "_")
    
    # 1. Main Results Directory
    results_dir = os.path.join(base_dir, "data", "results", "time", model_name_safe)
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Web Data Directory
    web_data_dir = os.path.join(base_dir, "web", "data")
    os.makedirs(web_data_dir, exist_ok=True)
    
    print(f"\nResults Directory: {results_dir}")
    print(f"Web Data Directory: {web_data_dir}\n")
    
    # Initialize experiment
    experiment = DiscountRateExperiment(
        amounts=AMOUNTS,
        delays=DELAYS,
        front_end_delays=FRONT_END_DELAYS,
        n_iterations=N_ITERATIONS,
        validation_fraction=0.1,
        print_progress=True,
        run_diagnostics=True
    )
    
    # Run full experiment
    analysis = experiment.run_full_experiment()
    
    # Generate visualizations
    experiment.generate_visualizations(output_dir=results_dir)
    
    # Save all results to the main results directory
    experiment.save_results(output_dir=results_dir)
    
    # Save specific web-chart data to the public web directory
    experiment.save_chart_data(output_dir=web_data_dir, analysis=analysis)
    
    # Generate final report
    experiment.generate_report(analysis, output_dir=results_dir)
    
    print("\n" + "="*70)
    print(" "*20 + "EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  📄 discount_rate_report.txt     - Comprehensive analysis report")
    print(f"  📊 discount_rate_results.csv    - Main results table")
    print(" "*20 + "EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print(f"  � {results_dir}")
    print(f"  � {web_data_dir}/time_experiment.json (for web visualization)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
