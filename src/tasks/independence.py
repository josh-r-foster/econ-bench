"""
Marschak-Machina Triangle Experiment: Testing EU Independence Axiom on LLM Preferences
(OpenAI API Version)

This script implements a comprehensive non-parametric mapping of an LLM's indifference
curves in the probability simplex to test whether the Independence axiom of Expected
Utility Theory holds for language model preferences.

Key Features:
- Phase 1: Fixed monetary outcomes and MM Triangle coordinate system
- Phase 2: 10-iteration bisection algorithm for high-precision indifference point elicitation
- Phase 3: Grid-based mapping across 50+ reference lotteries
- Phase 4: Slope analysis, fanning detection, and quadratic utility estimation
"""

from datetime import datetime
from tqdm import tqdm
import argparse
import sys
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to sys.path (to resolve 'src')
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(base_dir)

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
# 1. Configuration & Model Loading
# -------------------------------------------------------------



# -------------------------------------------------------------
# 2. Fixed Outcomes (Phase 1 Setup)
# -------------------------------------------------------------

# The three fixed monetary outcomes for the MM Triangle
X_LOW = 0      # $0 (worst outcome)
X_MID = 500     # $500 (middle outcome)
X_HIGH = 1000   # $1000 (best outcome)

# -------------------------------------------------------------
# 3. Data Structures
# -------------------------------------------------------------

@dataclass
class TrianglePoint:
    """A point in the Marschak-Machina Triangle
    
    Coordinates:
    - p_L: Probability of Low outcome ($0)
    - p_H: Probability of High outcome ($1000)
    - p_M: Probability of Middle outcome ($500) = 1 - p_L - p_H
    
    The triangle constraint: p_L + p_H <= 1, with p_L, p_H >= 0
    """
    p_L: float  # Probability of Low ($0)
    p_H: float  # Probability of High ($1000)
    
    def __post_init__(self):
        # Validate triangle constraints
        assert 0 <= self.p_L <= 1, f"p_L must be in [0,1], got {self.p_L}"
        assert 0 <= self.p_H <= 1, f"p_H must be in [0,1], got {self.p_H}"
        assert self.p_L + self.p_H <= 1 + 1e-9, f"p_L + p_H must be <= 1, got {self.p_L + self.p_H}"
    
    @property
    def p_M(self) -> float:
        """Probability of Middle outcome ($500)"""
        return 1.0 - self.p_L - self.p_H
    
    @property
    def expected_value(self) -> float:
        """Expected monetary value of this lottery"""
        return self.p_L * X_LOW + self.p_M * X_MID + self.p_H * X_HIGH
    
    def __repr__(self):
        return f"TrianglePoint(p_L={self.p_L:.3f}, p_H={self.p_H:.3f}, p_M={self.p_M:.3f})"
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.p_L, self.p_H)


@dataclass
class BisectionChoice:
    """Record of a single binary choice in the bisection routine"""
    iteration: int
    midpoint: float
    reference_lottery: str
    axis_lottery: str
    choice: str  # "A" (reference) or "B" (axis)
    raw_response: str
    lower_bound: float
    upper_bound: float


@dataclass
class BisectionResult:
    """Complete result of one bisection routine"""
    reference_point: TrianglePoint
    indifference_value: float  # The converged value on the axis
    axis: str  # "Y" (mixing High & Middle) or "X" (mixing Low & Middle)
    n_iterations: int
    choice_history: List[BisectionChoice]
    final_precision: float  # upper - lower at termination
    swap_order: bool = False  # True if axis lottery was presented as Option A
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def indifference_point(self) -> TrianglePoint:
        """The equivalent point on the axis"""
        if self.axis == "Y":
            # Y-axis: p_L = 0, varying p_H
            return TrianglePoint(p_L=0.0, p_H=self.indifference_value)
        else:
            # X-axis: p_H = 0, varying p_L
            return TrianglePoint(p_L=self.indifference_value, p_H=0.0)


@dataclass
class MonotonicityCheck:
    """Result of a dominance/monotonicity test"""
    dominant: TrianglePoint  # The objectively better lottery
    dominated: TrianglePoint  # The objectively worse lottery
    choice: str  # "DOMINANT" or "DOMINATED" or "FAILED_PARSE"
    passed: bool  # True if agent chose dominant
    raw_response: str
    ev_dominant: float
    ev_dominated: float


@dataclass
class TransitivityCheck:
    """Result of a transitivity test between two reference points"""
    point_a: TrianglePoint
    point_b: TrianglePoint
    indiff_a: float  # A's indifference value
    indiff_b: float  # B's indifference value
    predicted_preference: str  # "A" if indiff_a > indiff_b (for Y-axis), else "B"
    actual_preference: str  # What the agent actually chose
    consistent: bool  # True if predicted == actual
    raw_response: str


@dataclass
class IndifferenceCurve:
    """A traced indifference curve in the MM Triangle"""
    axis_value: float  # The Y or X intercept defining this curve
    axis: str  # "Y" or "X"
    points: List[TrianglePoint]
    
    @property
    def slopes(self) -> List[float]:
        """Calculate slopes between adjacent points"""
        slopes = []
        for i in range(len(self.points) - 1):
            p1, p2 = self.points[i], self.points[i + 1]
            dp_L = p2.p_L - p1.p_L
            dp_H = p2.p_H - p1.p_H
            if abs(dp_L) > 1e-9:
                slopes.append(dp_H / dp_L)
            else:
                slopes.append(float('inf'))
        return slopes


# -------------------------------------------------------------
# 4. Model Interface
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

class MMTrianglePrompts:
    """Prompts for the MM Triangle experiment"""
    
    @staticmethod
    def binary_choice(ref: TrianglePoint, axis_value: float, axis: str, 
                      swap_order: bool = False) -> str:
        """Create binary choice prompt between reference lottery and axis lottery
        
        Args:
            ref: Reference lottery point
            axis_value: Value on the axis (p_H for Y-axis, p_L for X-axis)
            axis: "Y" or "X"
            swap_order: If True, present axis lottery as A and reference as B
        """
        # Reference lottery description
        ref_parts = []
        if ref.p_H > 0:
            ref_parts.append(f"{ref.p_H*100:.1f}% chance of ${X_HIGH}")
        if ref.p_M > 0:
            ref_parts.append(f"{ref.p_M*100:.1f}% chance of ${X_MID}")
        if ref.p_L > 0:
            ref_parts.append(f"{ref.p_L*100:.1f}% chance of ${X_LOW}")
        ref_desc = ", ".join(ref_parts)
        
        # Axis lottery description
        if axis == "Y":
            # Y-axis: mixing High and Middle (no Low)
            axis_parts = []
            if axis_value > 0:
                axis_parts.append(f"{axis_value*100:.1f}% chance of ${X_HIGH}")
            if (1 - axis_value) > 0:
                axis_parts.append(f"{(1-axis_value)*100:.1f}% chance of ${X_MID}")
            axis_desc = ", ".join(axis_parts)
        else:
            # X-axis: mixing Low and Middle (no High)
            axis_parts = []
            if (1 - axis_value) > 0:
                axis_parts.append(f"{(1-axis_value)*100:.1f}% chance of ${X_MID}")
            if axis_value > 0:
                axis_parts.append(f"{axis_value*100:.1f}% chance of ${X_LOW}")
            axis_desc = ", ".join(axis_parts)
        
        # Swap order if requested (to detect positional bias)
        if swap_order:
            option_a_desc = axis_desc
            option_b_desc = ref_desc
        else:
            option_a_desc = ref_desc
            option_b_desc = axis_desc
        
        return f"""You must choose between two lotteries. Which do you prefer?

Option A: {option_a_desc}
Option B: {option_b_desc}

Respond with only the letter "A" or "B".

Answer:"""
    
    @staticmethod
    def explain_preference(ref: TrianglePoint, axis_value: float, axis: str, 
                           choice: str) -> str:
        """Ask for explanation of preference (optional, for validation)"""
        ref_desc = f"({ref.p_H*100:.0f}% ${X_HIGH}, {ref.p_M*100:.0f}% ${X_MID}, {ref.p_L*100:.0f}% ${X_LOW})"
        
        if axis == "Y":
            axis_desc = f"({axis_value*100:.0f}% ${X_HIGH}, {(1-axis_value)*100:.0f}% ${X_MID})"
        else:
            axis_desc = f"({(1-axis_value)*100:.0f}% ${X_MID}, {axis_value*100:.0f}% ${X_LOW})"
        
        return f"""You chose Option {choice}.

Option A was: {ref_desc}
Option B was: {axis_desc}

Briefly explain your reasoning in 1-2 sentences.

Explanation:"""


# -------------------------------------------------------------
# 6. Bisection Algorithm (Phase 2)
# -------------------------------------------------------------

class BisectionElicitor:
    """Implements the bisection routine for finding indifference points"""
    
    def __init__(self, n_iterations: int = 10, print_progress: bool = False,
                 print_interactions: bool = True):
        self.n_iterations = n_iterations
        self.print_progress = print_progress
        self.print_interactions = print_interactions
    
    def find_indifference_point(self, reference: TrianglePoint, 
                                 axis: str = "Y", swap_order: bool = False) -> BisectionResult:
        """
        Find the point on the specified axis that is indifferent to the reference lottery.
        
        Args:
            reference: The reference lottery in the MM Triangle
            axis: "Y" (mixing High & Middle) or "X" (mixing Low & Middle)
            swap_order: If True, present axis lottery as Option A (to detect positional bias)
        
        Returns:
            BisectionResult with the converged indifference value
        """
        lower = 0.0
        upper = 1.0
        choice_history = []
        
        for iteration in range(self.n_iterations):
            midpoint = (lower + upper) / 2
            
            # Create prompt (with optional order swap)
            prompt = MMTrianglePrompts.binary_choice(reference, midpoint, axis, swap_order)
            
            # Get response
            response = generate_response(prompt, print_interaction=self.print_interactions)
            raw_choice = parse_ab_choice(response)
            
            # Print parsed choice for verification
            if self.print_interactions:
                print(f"📋 PARSED CHOICE: {raw_choice if raw_choice else 'FAILED TO PARSE'}")
                if swap_order:
                    print(f"   (Order swapped: A=axis, B=reference)")
                print("─"*70 + "\n")
            
            if raw_choice is None:
                # If parsing failed, default to "A"
                if self.print_progress:
                    print(f"  ⚠️ Could not parse choice from: {response[:50]}...")
                raw_choice = "A"
            
            # Convert raw choice to semantic choice (ref vs axis)
            # When swap_order=True: A=axis, B=reference
            # When swap_order=False: A=reference, B=axis
            if swap_order:
                # Swapped: A means axis preferred, B means reference preferred
                chose_reference = (raw_choice == "B")
            else:
                # Normal: A means reference preferred, B means axis preferred
                chose_reference = (raw_choice == "A")
            
            # Use semantic labeling for the choice record
            semantic_choice = "REF" if chose_reference else "AXIS"
            
            # Record choice
            # Record choice
            ref_desc = f"({reference.p_H:.1%} ${X_HIGH}, {reference.p_M:.1%} ${X_MID}, {reference.p_L:.1%} ${X_LOW})"
            if axis == "Y":
                axis_desc = f"({midpoint:.1%} ${X_HIGH}, {1-midpoint:.1%} ${X_MID})"
            else:
                axis_desc = f"({1-midpoint:.1%} ${X_MID}, {midpoint:.1%} ${X_LOW})"
            
            choice_record = BisectionChoice(
                iteration=iteration + 1,
                midpoint=midpoint,
                reference_lottery=ref_desc,
                axis_lottery=axis_desc,
                choice=semantic_choice,  # Now stores REF/AXIS instead of A/B
                raw_response=response[:200],
                lower_bound=lower,
                upper_bound=upper
            )
            choice_history.append(choice_record)
            
            # Update bounds based on SEMANTIC choice (reference vs axis)
            # Y-axis: higher value = more $100 = BETTER
            # X-axis: higher value = more $0 = WORSE
            if axis == "Y":
                if chose_reference:
                    # Reference preferred, axis lottery was too bad (not enough $100)
                    lower = midpoint
                else:
                    # Axis lottery preferred, it was too good (too much $100)
                    upper = midpoint
            else:  # axis == "X"
                if chose_reference:
                    # Reference preferred, axis lottery was too bad (too much $0)
                    upper = midpoint
                else:
                    # Axis lottery preferred, it was too good (not enough $0)
                    lower = midpoint
            
            if self.print_progress:
                print(f"  Iteration {iteration+1}: midpoint={midpoint:.4f}, "
                      f"choice={semantic_choice}, bounds=[{lower:.4f}, {upper:.4f}]")
        
        # Final indifference value is the midpoint of remaining interval
        indifference_value = (lower + upper) / 2
        
        return BisectionResult(
            reference_point=reference,
            indifference_value=indifference_value,
            axis=axis,
            n_iterations=self.n_iterations,
            choice_history=choice_history,
            final_precision=upper - lower,
            swap_order=swap_order
        )


# -------------------------------------------------------------
# 7. Grid Generation (Phase 3)
# -------------------------------------------------------------

def generate_triangle_grid(n_divisions: int = 10) -> List[TrianglePoint]:
    """
    Generate a grid of points inside the MM Triangle.
    
    The triangle has vertices at:
    - (0, 0): 100% Middle (${X_MID})
    - (1, 0): 100% Low (${X_LOW})
    - (0, 1): 100% High (${X_HIGH})
    
    Args:
        n_divisions: Number of divisions along each axis
    
    Returns:
        List of TrianglePoint objects forming a triangular grid
    """
    points = []
    step = 1.0 / n_divisions
    
    for i in range(n_divisions + 1):
        for j in range(n_divisions + 1 - i):
            p_L = i * step
            p_H = j * step
            # Skip vertices (extreme lotteries)
            if (p_L == 1 and p_H == 0) or (p_L == 0 and p_H == 1):
                continue
            points.append(TrianglePoint(p_L=p_L, p_H=p_H))
    
    return points


def classify_lottery(point: TrianglePoint) -> str:
    """
    Classify a lottery as 'better' or 'worse' than the certain Middle outcome.
    
    Better: Should be mapped to Y-axis (mixing High & Middle)
    Worse: Should be mapped to X-axis (mixing Low & Middle)
    
    Using Expected Value as the classifier (for EU maximizer, this is exact)
    """
    ev_middle = X_MID  
    if point.expected_value >= ev_middle:
        return "better"  # Map to Y-axis
    else:
        return "worse"   # Map to X-axis


# -------------------------------------------------------------
# 8. Main Experiment Runner
# -------------------------------------------------------------

class MMTriangleExperiment:
    """Main experiment class for the MM Triangle elicitation"""
    
    def __init__(self, n_divisions: int = 7, n_iterations: int = 10,
                 validation_fraction: float = 0.1, print_progress: bool = True,
                 run_diagnostics: bool = True):
        """
        Args:
            n_divisions: Grid density (7 gives ~36 interior points)
            n_iterations: Bisection iterations per point (10 gives ~0.1% precision)
            validation_fraction: Fraction of points to re-test for consistency
            print_progress: Whether to print progress updates
            run_diagnostics: Whether to run monotonicity/transitivity/bidirectional checks
        """
        self.n_divisions = n_divisions
        self.n_iterations = n_iterations
        self.validation_fraction = validation_fraction
        self.print_progress = print_progress
        self.run_diagnostics = run_diagnostics
        
        self.elicitor = BisectionElicitor(n_iterations=n_iterations, 
                                          print_progress=print_progress,
                                          print_interactions=PRINT_INTERACTIONS)
        self.results: List[BisectionResult] = []
        self.validation_results: List[Tuple[BisectionResult, BisectionResult]] = []
        
        # New diagnostic storage
        self.monotonicity_checks: List[MonotonicityCheck] = []
        self.transitivity_checks: List[TransitivityCheck] = []
        self.bidirectional_results: List[Tuple[BisectionResult, BisectionResult]] = []
    
    def _compare_lotteries(self, lottery_a: TrianglePoint, lottery_b: TrianglePoint) -> Tuple[str, str]:
        """
        Directly compare two lotteries and return the choice.
        
        Returns:
            Tuple of (choice: "A" or "B", raw_response: str)
        """
        # Build descriptions
        a_parts = []
        if lottery_a.p_H > 0:
            a_parts.append(f"{lottery_a.p_H*100:.1f}% chance of ${X_HIGH}")
        if lottery_a.p_M > 0:
            a_parts.append(f"{lottery_a.p_M*100:.1f}% chance of ${X_MID}")
        if lottery_a.p_L > 0:
            a_parts.append(f"{lottery_a.p_L*100:.1f}% chance of ${X_LOW}")
        a_desc = ", ".join(a_parts)
        
        b_parts = []
        if lottery_b.p_H > 0:
            b_parts.append(f"{lottery_b.p_H*100:.1f}% chance of ${X_HIGH}")
        if lottery_b.p_M > 0:
            b_parts.append(f"{lottery_b.p_M*100:.1f}% chance of ${X_MID}")
        if lottery_b.p_L > 0:
            b_parts.append(f"{lottery_b.p_L*100:.1f}% chance of ${X_LOW}")
        b_desc = ", ".join(b_parts)
        
        prompt = f"""You must choose between two lotteries. Which do you prefer?

Option A: {a_desc}
Option B: {b_desc}

Respond with only the letter "A" or "B".

Answer:"""
        
        response = generate_response(prompt, print_interaction=PRINT_INTERACTIONS)
        choice = parse_ab_choice(response)
        if choice is None:
            choice = "A"  # Default
        return choice, response
    
    def run_monotonicity_checks(self, n_checks: int = 5) -> List[MonotonicityCheck]:
        """
        Run dominance/monotonicity sanity checks.
        
        These test whether the agent prefers strictly better lotteries
        (more $100, less $0) over strictly worse ones.
        """
        print("\nMONOTONICITY CHECKS: Dominance Tests")
        print("-"*70)
        
        # Define clearly dominant/dominated pairs
        test_pairs = [
            # (dominant, dominated) - dominant has higher EV by stochastic dominance
            (TrianglePoint(p_L=0.1, p_H=0.6), TrianglePoint(p_L=0.3, p_H=0.4)),  # EV: 75 vs 55
            (TrianglePoint(p_L=0.0, p_H=0.8), TrianglePoint(p_L=0.2, p_H=0.6)),  # EV: 90 vs 70
            (TrianglePoint(p_L=0.2, p_H=0.4), TrianglePoint(p_L=0.4, p_H=0.2)),  # EV: 60 vs 40
            (TrianglePoint(p_L=0.0, p_H=0.5), TrianglePoint(p_L=0.0, p_H=0.3)),  # EV: 75 vs 65 (pure Y-axis)
            (TrianglePoint(p_L=0.3, p_H=0.0), TrianglePoint(p_L=0.5, p_H=0.0)),  # EV: 35 vs 25 (pure X-axis)
        ][:n_checks]
        
        for dominant, dominated in tqdm(test_pairs, desc="Monotonicity tests"):
            choice, response = self._compare_lotteries(dominant, dominated)
            
            # A = dominant, B = dominated, so we expect A
            passed = (choice == "A")
            
            check = MonotonicityCheck(
                dominant=dominant,
                dominated=dominated,
                choice="DOMINANT" if choice == "A" else "DOMINATED",
                passed=passed,
                raw_response=response[:200],
                ev_dominant=dominant.expected_value,
                ev_dominated=dominated.expected_value
            )
            self.monotonicity_checks.append(check)
            
            status = "✓ PASS" if passed else "✗ FAIL"
            if self.print_progress:
                tqdm.write(f"  {status}: {dominant} vs {dominated}")
        
        n_passed = sum(1 for c in self.monotonicity_checks if c.passed)
        print(f"\n  Results: {n_passed}/{len(self.monotonicity_checks)} passed")
        
        return self.monotonicity_checks
    
    def run_transitivity_checks(self, n_checks: int = 10) -> List[TransitivityCheck]:
        """
        Run transitivity checks by comparing reference points with similar indifference values.
        
        If A maps to higher indifference value than B (on same axis), then A should be preferred to B.
        """
        print("\nTRANSITIVITY CHECKS: Indirect Preference Consistency")
        print("-"*70)
        
        if len(self.results) < 2:
            print("  Not enough results for transitivity checks")
            return []
        
        # Group by axis and sample pairs
        y_results = [r for r in self.results if r.axis == "Y"]
        x_results = [r for r in self.results if r.axis == "X"]
        
        pairs_to_test = []
        
        # Sample from Y-axis results (more indiff value = better)
        if len(y_results) >= 2:
            sorted_y = sorted(y_results, key=lambda r: r.indifference_value)
            for i in range(min(n_checks // 2, len(sorted_y) - 1)):
                # Compare adjacent pairs (should have clear ordering)
                pairs_to_test.append((sorted_y[i+1], sorted_y[i], "Y"))  # higher first
        
        # Sample from X-axis results (more indiff value = worse)
        if len(x_results) >= 2:
            sorted_x = sorted(x_results, key=lambda r: r.indifference_value)
            for i in range(min(n_checks // 2, len(sorted_x) - 1)):
                # For X-axis, lower indiff value = better
                pairs_to_test.append((sorted_x[i], sorted_x[i+1], "X"))  # lower first
        
        for result_a, result_b, axis in tqdm(pairs_to_test[:n_checks], desc="Transitivity tests"):
            point_a = result_a.reference_point
            point_b = result_b.reference_point
            
            # Expected: A should be preferred (it maps to "better" axis value)
            choice, response = self._compare_lotteries(point_a, point_b)
            consistent = (choice == "A")
            
            check = TransitivityCheck(
                point_a=point_a,
                point_b=point_b,
                indiff_a=result_a.indifference_value,
                indiff_b=result_b.indifference_value,
                predicted_preference="A",
                actual_preference=choice,
                consistent=consistent,
                raw_response=response[:200]
            )
            self.transitivity_checks.append(check)
            
            status = "✓" if consistent else "✗"
            if self.print_progress:
                tqdm.write(f"  {status}: A({result_a.indifference_value:.3f}) vs B({result_b.indifference_value:.3f}) → {choice}")
        
        n_consistent = sum(1 for c in self.transitivity_checks if c.consistent)
        print(f"\n  Results: {n_consistent}/{len(self.transitivity_checks)} consistent")
        
        return self.transitivity_checks
    
    def run_bidirectional_bisection(self, n_samples: int = 5) -> List[Tuple[BisectionResult, BisectionResult]]:
        """
        Re-run bisection with swapped A/B order for a sample of X-axis points.
        
        This detects positional bias - if results differ significantly, the agent
        has a systematic preference for Option A or B regardless of content.
        """
        print("\nBIDIRECTIONAL BISECTION: Positional Bias Detection")
        print("-"*70)
        
        # Focus on X-axis results (where we saw pathological behavior)
        x_results = [r for r in self.results if r.axis == "X" and not r.swap_order]
        
        if len(x_results) == 0:
            print("  No X-axis results to test")
            return []
        
        # Sample points to re-test with swapped order
        sample_indices = np.random.choice(len(x_results), min(n_samples, len(x_results)), replace=False)
        
        for idx in tqdm(sample_indices, desc="Bidirectional bisection"):
            original = x_results[idx]
            point = original.reference_point
            
            # Re-run with swapped order
            swapped_result = self.elicitor.find_indifference_point(point, axis="X", swap_order=True)
            self.bidirectional_results.append((original, swapped_result))
            
            diff = abs(original.indifference_value - swapped_result.indifference_value)
            if self.print_progress:
                tqdm.write(f"  {point}: normal={original.indifference_value:.4f}, "
                          f"swapped={swapped_result.indifference_value:.4f}, diff={diff:.4f}")
        
        # Calculate bias metrics
        if self.bidirectional_results:
            diffs = [abs(o.indifference_value - s.indifference_value) for o, s in self.bidirectional_results]
            mean_diff = np.mean(diffs)
            print(f"\n  Mean difference: {mean_diff:.4f}")
            print(f"  Positional bias: {'DETECTED' if mean_diff > 0.1 else 'MINIMAL'}")
        
        return self.bidirectional_results


    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete 4-phase experiment"""
        print("\n" + "="*70)
        print(" "*10 + "MARSCHAK-MACHINA TRIANGLE EXPERIMENT")
        print(" "*15 + f"Model: {model_id}")
        print("="*70 + "\n")
        
        # Phase 1: Setup
        print("PHASE 1: Setup")
        print("-"*70)
        print(f"  Outcomes: ${X_LOW} (Low), ${X_MID} (Middle), ${X_HIGH} (High)")
        print(f"  Grid divisions: {self.n_divisions}")
        print(f"  Bisection iterations: {self.n_iterations}")
        print(f"  Precision: ~{1/2**self.n_iterations:.4f}")
        
        # Generate grid
        grid_points = generate_triangle_grid(self.n_divisions)
        # Filter out the origin (certain Middle) as it's not interesting
        grid_points = [p for p in grid_points if not (p.p_L == 0 and p.p_H == 0)]
        
        print(f"  Grid points to test: {len(grid_points)}")
        print()
        
        # Phase 2 & 3: Elicitation
        print("PHASE 2-3: Bisection Elicitation")
        print("-"*70)
        
        n_better = sum(1 for p in grid_points if classify_lottery(p) == "better")
        n_worse = len(grid_points) - n_better
        print(f"  'Better-than-Middle' lotteries (→ Y-axis): {n_better}")
        print(f"  'Worse-than-Middle' lotteries (→ X-axis): {n_worse}")
        print()
        
        # Run bisection for each grid point
        for point in tqdm(grid_points, desc="Eliciting preferences"):
            classification = classify_lottery(point)
            axis = "Y" if classification == "better" else "X"
            
            result = self.elicitor.find_indifference_point(point, axis=axis)
            self.results.append(result)
            
            if self.print_progress:
                tqdm.write(f"  {point} → {axis}-axis: {result.indifference_value:.4f}")
        
        print()
        
        # Validation: Re-test subset
        print("VALIDATION: Consistency Check")
        print("-"*70)
        n_validation = max(1, int(len(grid_points) * self.validation_fraction))
        validation_indices = np.random.choice(len(self.results), n_validation, replace=False)
        
        for idx in tqdm(validation_indices, desc="Validation checks"):
            original_result = self.results[idx]
            point = original_result.reference_point
            axis = original_result.axis
            
            # Re-run bisection
            retest_result = self.elicitor.find_indifference_point(point, axis=axis)
            self.validation_results.append((original_result, retest_result))
            
            deviation = abs(original_result.indifference_value - retest_result.indifference_value)
            if self.print_progress:
                tqdm.write(f"  {point}: original={original_result.indifference_value:.4f}, "
                          f"retest={retest_result.indifference_value:.4f}, dev={deviation:.4f}")
        
        # Calculate consistency metrics
        deviations = [abs(o.indifference_value - r.indifference_value) 
                     for o, r in self.validation_results]
        mean_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        
        print(f"\n  Mean deviation: {mean_deviation:.4f}")
        print(f"  Max deviation: {max_deviation:.4f}")
        print(f"  Consistency: {'GOOD' if mean_deviation < 0.05 else 'MODERATE' if mean_deviation < 0.1 else 'POOR'}")
        print()
        
        # NEW DIAGNOSTIC PHASES (if enabled)
        if self.run_diagnostics:
            # Monotonicity checks
            self.run_monotonicity_checks(n_checks=5)
            
            # Transitivity checks (requires main results)
            self.run_transitivity_checks(n_checks=10)
            
            # Bidirectional bisection (for X-axis points)
            self.run_bidirectional_bisection(n_samples=5)
        
        # Phase 4: Analysis
        print("\nPHASE 4: Analysis")
        print("-"*70)
        
        analysis = self.analyze_results()
        
        return analysis
    
    def analyze_results(self) -> Dict[str, Any]:
        """Perform Phase 4 analysis on collected data"""
        
        # Group results by indifference value (to trace curves)
        y_axis_results = [r for r in self.results if r.axis == "Y"]
        x_axis_results = [r for r in self.results if r.axis == "X"]
        
        # Calculate local slopes
        slopes_data = self._calculate_local_slopes()
        
        # Test for parallelism (EU vs fanning)
        parallelism_test = self._test_parallelism(slopes_data)
        
        # Fit quadratic utility model
        quadratic_fit = self._fit_quadratic_utility()
        
        analysis = {
            "n_points_tested": len(self.results),
            "n_y_axis_lotteries": len(y_axis_results),
            "n_x_axis_lotteries": len(x_axis_results),
            "slopes": slopes_data,
            "parallelism_test": parallelism_test,
            "quadratic_fit": quadratic_fit,
            "validation": {
                "n_retests": len(self.validation_results),
                "mean_deviation": float(np.mean([abs(o.indifference_value - r.indifference_value) 
                                                  for o, r in self.validation_results])) if self.validation_results else 0.0,
                "max_deviation": float(np.max([abs(o.indifference_value - r.indifference_value) 
                                                for o, r in self.validation_results])) if self.validation_results else 0.0
            },
            # Diagnostic results
            "diagnostics": {
                "monotonicity": {
                    "n_checks": len(self.monotonicity_checks),
                    "n_passed": sum(1 for c in self.monotonicity_checks if c.passed),
                    "pass_rate": sum(1 for c in self.monotonicity_checks if c.passed) / len(self.monotonicity_checks) if self.monotonicity_checks else 1.0
                },
                "transitivity": {
                    "n_checks": len(self.transitivity_checks),
                    "n_consistent": sum(1 for c in self.transitivity_checks if c.consistent),
                    "consistency_rate": sum(1 for c in self.transitivity_checks if c.consistent) / len(self.transitivity_checks) if self.transitivity_checks else 1.0
                },
                "bidirectional": {
                    "n_samples": len(self.bidirectional_results),
                    "mean_difference": float(np.mean([abs(o.indifference_value - s.indifference_value) for o, s in self.bidirectional_results])) if self.bidirectional_results else 0.0,
                    "positional_bias_detected": (np.mean([abs(o.indifference_value - s.indifference_value) for o, s in self.bidirectional_results]) > 0.1) if self.bidirectional_results else False
                }
            }
        }
        
        # Print summary
        print(f"  Points tested: {analysis['n_points_tested']}")
        print(f"  Y-axis mappings: {analysis['n_y_axis_lotteries']}")
        print(f"  X-axis mappings: {analysis['n_x_axis_lotteries']}")
        print()
        
        print("  PARALLELISM TEST (Independence Axiom):")
        print(f"    Mean slope: {parallelism_test['mean_slope']:.4f}")
        print(f"    Slope std dev: {parallelism_test['slope_std']:.4f}")
        print(f"    Fanning pattern: {parallelism_test['fanning_pattern']}")
        print(f"    EU violation: {'YES' if parallelism_test['eu_violated'] else 'NO'}")
        print()
        
        print("  QUADRATIC UTILITY FIT:")
        for name, value in quadratic_fit.items():
            if isinstance(value, float):
                print(f"    {name}: {value:.6f}")
            else:
                print(f"    {name}: {value}")
        print()
        
        # Print diagnostics summary
        if self.run_diagnostics:
            diag = analysis["diagnostics"]
            print("  DIAGNOSTICS SUMMARY:")
            print(f"    Monotonicity: {diag['monotonicity']['n_passed']}/{diag['monotonicity']['n_checks']} passed")
            print(f"    Transitivity: {diag['transitivity']['n_consistent']}/{diag['transitivity']['n_checks']} consistent")
            print(f"    Positional bias: {'DETECTED' if diag['bidirectional']['positional_bias_detected'] else 'NOT DETECTED'}")
            
            # Calculate overall rationality score
            mono_score = diag['monotonicity']['pass_rate']
            trans_score = diag['transitivity']['consistency_rate']
            bias_penalty = 0.2 if diag['bidirectional']['positional_bias_detected'] else 0.0
            rationality_score = (mono_score + trans_score) / 2 - bias_penalty
            print(f"    Rationality score: {rationality_score:.2f} / 1.00")
            analysis["diagnostics"]["rationality_score"] = rationality_score
        print()
        
        return analysis
    
    def _calculate_local_slopes(self) -> List[Dict]:
        """Calculate local slopes throughout the triangle"""
        slopes_data = []
        
        for result in self.results:
            ref = result.reference_point
            indiff = result.indifference_point
            
            # Skip points that are already on the target axis (or very close)
            # because the "movement" is just bisection noise
            if result.axis == "Y" and ref.p_L < 0.01:
                slope = float('nan')
            elif result.axis == "X" and ref.p_H < 0.01:
                slope = float('nan')
            else:
                dp_L = indiff.p_L - ref.p_L
                dp_H = indiff.p_H - ref.p_H
                
                if abs(dp_L) > 1e-9:
                    slope = dp_H / dp_L
                else:
                    slope = float('inf') if dp_H > 0 else float('-inf') if dp_H < 0 else 0
            
            slopes_data.append({
                "p_L": ref.p_L,
                "p_H": ref.p_H,
                "p_M": ref.p_M,
                "indiff_value": result.indifference_value,
                "axis": result.axis,
                "slope": slope,
                "ev": ref.expected_value
            })
        
        return slopes_data
    
    def _test_parallelism(self, slopes_data: List[Dict]) -> Dict:
        """Test whether indifference curves are parallel (EU) or fanning"""
        
        # Filter finite slopes
        finite_slopes = [s["slope"] for s in slopes_data if np.isfinite(s["slope"])]
        
        if len(finite_slopes) < 2:
            return {
                "mean_slope": 0.0,
                "slope_std": 0.0,
                "fanning_pattern": "INSUFFICIENT_DATA",
                "eu_violated": False
            }
        
        mean_slope = np.mean(finite_slopes)
        slope_std = np.std(finite_slopes)
        
        # Coefficient of variation
        cv = slope_std / abs(mean_slope) if abs(mean_slope) > 1e-9 else float('inf')
        
        # Check for systematic fanning
        # Fanning out: slopes increase as we move toward (0, 1) corner
        # Fanning in: slopes decrease as we move toward (0, 1) corner
        
        # Regress slope on position in triangle
        p_H_values = np.array([s["p_H"] for s in slopes_data if np.isfinite(s["slope"])])
        slope_values = np.array(finite_slopes)
        
        if len(p_H_values) > 2:
            correlation = np.corrcoef(p_H_values, slope_values)[0, 1]
        else:
            correlation = 0.0
        
        # Determine fanning pattern
        if cv < 0.15:  # Low variation suggests parallel lines (EU)
            fanning_pattern = "PARALLEL (EU holds)"
            eu_violated = False
        elif correlation > 0.3:
            fanning_pattern = "FANNING_OUT (Common EU violation)"
            eu_violated = True
        elif correlation < -0.3:
            fanning_pattern = "FANNING_IN (Rare)"
            eu_violated = True
        else:
            fanning_pattern = "IRREGULAR (Non-systematic deviation)"
            eu_violated = True
        
        return {
            "mean_slope": float(mean_slope),
            "slope_std": float(slope_std),
            "coefficient_of_variation": float(cv),
            "slope_position_correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "fanning_pattern": fanning_pattern,
            "eu_violated": eu_violated
        }
    
    def _fit_quadratic_utility(self) -> Dict:
        """
        Fit a quadratic utility function:
        V(p) = Σ α_i p_i + (1/2) Σ Σ β_ij p_i p_j
        
        For three outcomes (L, M, H), this becomes:
        V = α_L*p_L + α_M*p_M + α_H*p_H + 
            (1/2)*(β_LL*p_L² + β_MM*p_M² + β_HH*p_H² + 
                   2*β_LM*p_L*p_M + 2*β_LH*p_L*p_H + 2*β_MH*p_M*p_H)
        """
        
        # Collect data: reference point ≈ indifference point (same utility)
        data = []
        for result in self.results:
            ref = result.reference_point
            indiff = result.indifference_point
            # These two should have the same utility
            data.append((ref.p_L, ref.p_M, ref.p_H, indiff.p_L, indiff.p_M, indiff.p_H))
        
        if len(data) < 6:
            return {"status": "INSUFFICIENT_DATA", "beta_norm": 0.0, "is_eu": False}
        
        data = np.array(data)
        
        def quadratic_utility(p_L, p_M, p_H, params):
            """Calculate quadratic utility for given probabilities"""
            alpha_L, alpha_M, alpha_H = params[:3]
            beta_LL, beta_MM, beta_HH, beta_LM, beta_LH, beta_MH = params[3:]
            
            linear = alpha_L * p_L + alpha_M * p_M + alpha_H * p_H
            quadratic = 0.5 * (beta_LL * p_L**2 + beta_MM * p_M**2 + beta_HH * p_H**2 +
                               2 * beta_LM * p_L * p_M + 2 * beta_LH * p_L * p_H + 
                               2 * beta_MH * p_M * p_H)
            return linear + quadratic
        
        def loss(params):
            """Sum of squared differences between utilities of indifferent pairs"""
            total_loss = 0
            for row in data:
                u_ref = quadratic_utility(row[0], row[1], row[2], params)
                u_indiff = quadratic_utility(row[3], row[4], row[5], params)
                total_loss += (u_ref - u_indiff) ** 2
            return total_loss
        
        # Initial guess: linear utility (EU)
        # α_L = 0, α_M = 50, α_H = 100 (normalized values)
        # β = 0 (no quadratic terms)
        x0 = np.array([0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Minimize
        result = minimize(loss, x0, method='L-BFGS-B')
        
        params = result.x
        beta_params = params[3:]
        beta_norm = np.linalg.norm(beta_params)
        
        # EU holds if all beta ≈ 0
        is_eu = beta_norm < 0.05
        
        return {
            "status": "SUCCESS",
            "alpha_L": float(params[0]),
            "alpha_M": float(params[1]),
            "alpha_H": float(params[2]),
            "beta_LL": float(params[3]),
            "beta_MM": float(params[4]),
            "beta_HH": float(params[5]),
            "beta_LM": float(params[6]),
            "beta_LH": float(params[7]),
            "beta_MH": float(params[8]),
            "beta_norm": float(beta_norm),
            "is_eu": is_eu,
            "interpretation": "EU Maximizer" if is_eu else "EU Violator (Non-zero β terms)",
            "residual_loss": float(result.fun)
        }
    
    def generate_visualizations(self, output_dir: str = "."):
        """Generate all visualization plots"""
        
        print("\nGenerating visualizations...")
        print("-"*70)
        
        # 1. Triangle with grid points and mappings
        self._plot_triangle_grid(os.path.join(output_dir, "mm_triangle_grid.png"))
        
        # 2. Indifference curves
        self._plot_indifference_curves(os.path.join(output_dir, "indifference_curves.png"))
        
        # 3. Slope heatmap
        self._plot_slope_heatmap(os.path.join(output_dir, "slope_heatmap.png"))
        
        # 4. Slope vector field
        self._plot_slope_vectors(os.path.join(output_dir, "slope_vector_field.png"))
        
        # 5. EU comparison
        self._plot_eu_comparison(os.path.join(output_dir, "eu_comparison.png"))
        
        # 6. EU deviation plot (NEW)
        self._plot_eu_deviation(os.path.join(output_dir, "eu_deviation.png"))
        
        print()
    
    def _plot_triangle_grid(self, save_path: str):
        """Plot the MM Triangle with tested grid points"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw triangle boundary
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        # Plot grid points colored by axis mapping
        for result in self.results:
            ref = result.reference_point
            color = 'blue' if result.axis == "Y" else 'red'
            ax.scatter(ref.p_L, ref.p_H, c=color, s=100, alpha=0.7, zorder=5)
            
            # Draw arrow to indifference point
            indiff = result.indifference_point
            ax.annotate('', xy=(indiff.p_L, indiff.p_H), xytext=(ref.p_L, ref.p_H),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.3))
        
        # Mark vertices
        ax.scatter([0, 1, 0], [0, 0, 1], c='black', s=200, marker='s', zorder=10)
        ax.annotate(f'100% ${X_MID}', (0, 0), xytext=(-0.1, -0.05), fontsize=10)
        ax.annotate(f'100% ${X_LOW}', (1, 0), xytext=(1.02, -0.02), fontsize=10)
        ax.annotate(f'100% ${X_HIGH}', (0, 1), xytext=(-0.15, 1.02), fontsize=10)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel(r'$p_L$ (Probability of $0)', fontsize=12)
        ax.set_ylabel(r'$p_H$ (Probability of $100)', fontsize=12)
        ax.set_title('Marschak-Machina Triangle: Grid Points and Mappings', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(['Y-axis mapping (better)', 'X-axis mapping (worse)'], loc='upper right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _plot_indifference_curves(self, save_path: str):
        """Plot elicited indifference curves
        
        Each result maps a reference point to an indifference point on an axis.
        We draw lines connecting reference points to their axis intercepts,
        colored by indifference value to show approximate utility levels.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw triangle boundary
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        # 1. Plot Y-axis results (better-than-$50 lotteries -> Y-axis)
        y_results = [r for r in self.results if r.axis == "Y"]
        if y_results:
            # Color by indifference value (higher = more $100 equivalent)
            values = [r.indifference_value for r in y_results]
            norm = plt.Normalize(min(values), max(values))
            cmap = plt.cm.viridis
            
            for r in y_results:
                ref = r.reference_point
                indiff_pt = (0, r.indifference_value)  # Y-axis intercept
                color = cmap(norm(r.indifference_value))
                
                # Draw line from reference to axis intercept
                ax.plot([ref.p_L, indiff_pt[0]], [ref.p_H, indiff_pt[1]], 
                       '-', color=color, linewidth=1.5, alpha=0.6)
                # Mark reference point
                ax.scatter(ref.p_L, ref.p_H, c=[color], s=50, marker='o', 
                          edgecolors='white', linewidth=0.5, zorder=5)

        # 2. Plot X-axis results (worse-than-$50 lotteries -> X-axis)
        x_results = [r for r in self.results if r.axis == "X"]
        if x_results:
            # Color by indifference value (higher = more $0 equivalent = worse)
            values = [r.indifference_value for r in x_results]
            norm = plt.Normalize(min(values), max(values))
            cmap = plt.cm.plasma
            
            for r in x_results:
                ref = r.reference_point
                indiff_pt = (r.indifference_value, 0)  # X-axis intercept
                color = cmap(norm(r.indifference_value))
                
                # Draw line from reference to axis intercept
                ax.plot([ref.p_L, indiff_pt[0]], [ref.p_H, indiff_pt[1]], 
                       '-', color=color, linewidth=1.5, alpha=0.6)
                # Mark reference point
                ax.scatter(ref.p_L, ref.p_H, c=[color], s=50, marker='s', 
                          edgecolors='white', linewidth=0.5, zorder=5)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$p_L$ (Probability of $0)', fontsize=12)
        ax.set_ylabel(r'$p_H$ (Probability of $100)', fontsize=12)
        ax.set_title('Estimated Indifference Curves (Full Triangle)', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()

    def _plot_slope_heatmap(self, save_path: str):
        """Plot heatmap of local slopes"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        p_L_vals, p_H_vals, slopes = [], [], []
        for result in self.results:
            ref = result.reference_point
            indiff = result.indifference_point
            
            # Use improved slope calc logic here too for display
            if result.axis == "Y" and ref.p_L < 0.01: continue
            if result.axis == "X" and ref.p_H < 0.01: continue
            
            dp_L = indiff.p_L - ref.p_L
            dp_H = indiff.p_H - ref.p_H
            if abs(dp_L) > 0.01:
                slope = dp_H / dp_L
                if -10 < slope < 10:
                    p_L_vals.append(ref.p_L)
                    p_H_vals.append(ref.p_H)
                    slopes.append(slope)
        
        if slopes:
            scatter = ax.scatter(p_L_vals, p_H_vals, c=slopes, cmap='RdYlBu_r', s=200, edgecolors='black')
            plt.colorbar(scatter, ax=ax, label='Local Slope')
        
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$p_L$', fontsize=12)
        ax.set_ylabel(r'$p_H$', fontsize=12)
        ax.set_title('Slope Heatmap: Local Indifference Patterns', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()

    def _plot_slope_vectors(self, save_path: str):
        """Plot vector field comparing actual mappings to EU-predicted mappings
        
        IMPORTANT: This shows the direction from reference points to axis intercepts,
        NOT the direction along indifference curves. For a rational EU agent,
        the mapping direction depends on where the reference point is located.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Calculate EU-predicted and actual vectors for each result
        eu_vectors = []
        actual_vectors = []
        angular_deviations = []
        
        for result in self.results:
            ref = result.reference_point
            ref_ev = ref.expected_value
            
            # Calculate EU-predicted axis intercept
            if result.axis == "Y":
                eu_indiff = (ref_ev - 50) / 50  # p_H on Y-axis
                eu_indiff = max(0, min(1, eu_indiff))
                eu_target = (0, eu_indiff)
                actual_target = (0, result.indifference_value)
            else:
                eu_indiff = (50 - ref_ev) / 50  # p_L on X-axis  
                eu_indiff = max(0, min(1, eu_indiff))
                eu_target = (eu_indiff, 0)
                actual_target = (result.indifference_value, 0)
            
            # EU-predicted direction vector
            eu_dx = eu_target[0] - ref.p_L
            eu_dy = eu_target[1] - ref.p_H
            eu_mag = np.sqrt(eu_dx**2 + eu_dy**2)
            
            # Actual direction vector
            actual_dx = actual_target[0] - ref.p_L
            actual_dy = actual_target[1] - ref.p_H
            actual_mag = np.sqrt(actual_dx**2 + actual_dy**2)
            
            if eu_mag > 0.01 and actual_mag > 0.01:
                eu_vectors.append((ref, eu_dx/eu_mag, eu_dy/eu_mag, result.axis))
                actual_vectors.append((ref, actual_dx/actual_mag, actual_dy/actual_mag, result.axis))
                
                # Calculate angular deviation
                dot = (eu_dx/eu_mag) * (actual_dx/actual_mag) + (eu_dy/eu_mag) * (actual_dy/actual_mag)
                dot = max(-1, min(1, dot))  # Clamp for numerical stability
                angle_rad = np.arccos(dot)
                angular_deviations.append(np.degrees(angle_rad))
        
        # Left panel: EU-predicted mapping directions
        ax = axes[0]
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        for ref, dx, dy, axis in eu_vectors:
            scale = 0.06
            color = 'forestgreen' if axis == "Y" else 'darkorange'
            ax.arrow(ref.p_L, ref.p_H, dx * scale, dy * scale,
                    head_width=0.015, head_length=0.01,
                    fc=color, ec=color, alpha=0.7)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$p_L$ (Probability of $0)', fontsize=12)
        ax.set_ylabel(r'$p_H$ (Probability of $100)', fontsize=12)
        ax.set_title('EU Prediction: Expected Mapping Directions', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='forestgreen', label='Y-axis mappings'),
                          Patch(facecolor='darkorange', label='X-axis mappings')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Right panel: Actual mappings colored by deviation from EU
        ax = axes[1]
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        if angular_deviations:
            max_dev = max(angular_deviations) if max(angular_deviations) > 0 else 45
            norm = plt.Normalize(0, min(max_dev, 90))  # Cap at 90 degrees
            cmap = plt.cm.RdYlGn_r  # Green = low deviation, Red = high
            
            for i, (ref, dx, dy, axis) in enumerate(actual_vectors):
                scale = 0.06
                color = cmap(norm(angular_deviations[i]))
                ax.arrow(ref.p_L, ref.p_H, dx * scale, dy * scale,
                        head_width=0.015, head_length=0.01,
                        fc=color, ec=color, alpha=0.8)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Angular Deviation from EU (degrees)')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$p_L$ (Probability of $0)', fontsize=12)
        ax.set_ylabel(r'$p_H$ (Probability of $100)', fontsize=12)
        ax.set_title('Actual: Mapping Directions (colored by EU deviation)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        # Add mean deviation annotation
        if angular_deviations:
            mean_dev = np.mean(angular_deviations)
            ax.text(0.95, 0.05, f'Mean deviation: {mean_dev:.1f}°', 
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Mapping Direction Comparison: EU Prediction vs Actual', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _plot_eu_comparison(self, save_path: str):
        """Compare actual curves with EU prediction"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: EU prediction (parallel lines)
        ax = axes[0]
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        # For Risk Neutral EU: p_H = p_L + C
        for C in np.linspace(-0.8, 0.8, 9):
            start_pL = max(0, -C)
            end_pL = (1 - C) / 2
            
            if end_pL > start_pL:
                x = np.linspace(start_pL, end_pL, 100)
                y = x + C
                ax.plot(x, y, 'b-', linewidth=2, alpha=0.5)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$p_L$', fontsize=12)
        ax.set_ylabel(r'$p_H$', fontsize=12)
        ax.set_title('EU Prediction: Positive Slope (+1)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        # Right: Actual data
        ax = axes[1]
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        # 1. Plot Y-axis results - draw lines from reference to axis intercept
        y_results = [r for r in self.results if r.axis == "Y"]
        if y_results:
            values = [r.indifference_value for r in y_results]
            norm = plt.Normalize(min(values), max(values))
            cmap = plt.cm.viridis
            
            for r in y_results:
                ref = r.reference_point
                indiff_pt = (0, r.indifference_value)
                color = cmap(norm(r.indifference_value))
                ax.plot([ref.p_L, indiff_pt[0]], [ref.p_H, indiff_pt[1]], 
                       '-', color=color, linewidth=1.5, alpha=0.6)
                ax.scatter(ref.p_L, ref.p_H, c=[color], s=40, marker='o', 
                          edgecolors='white', linewidth=0.5, zorder=5)

        # 2. Plot X-axis results - draw lines from reference to axis intercept
        x_results = [r for r in self.results if r.axis == "X"]
        if x_results:
            values = [r.indifference_value for r in x_results]
            norm = plt.Normalize(min(values), max(values))
            cmap = plt.cm.plasma
            
            for r in x_results:
                ref = r.reference_point
                indiff_pt = (r.indifference_value, 0)
                color = cmap(norm(r.indifference_value))
                ax.plot([ref.p_L, indiff_pt[0]], [ref.p_H, indiff_pt[1]], 
                       '-', color=color, linewidth=1.5, alpha=0.6)
                ax.scatter(ref.p_L, ref.p_H, c=[color], s=40, marker='s', 
                          edgecolors='white', linewidth=0.5, zorder=5)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$p_L$', fontsize=12)
        ax.set_ylabel(r'$p_H$', fontsize=12)
        ax.set_title('Actual Data: Elicited Indifference Curves', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        plt.suptitle('Expected Utility Prediction vs Actual LLM Preferences', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _plot_eu_deviation(self, save_path: str):
        """Plot indifference mappings colored by deviation from EU prediction
        
        For each reference point, calculate what a risk-neutral EU agent would predict
        as the indifference value, then color the mapping by how much the actual
        result deviates from this prediction.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw triangle boundary
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        deviations = []
        plot_data = []
        
        for r in self.results:
            ref = r.reference_point
            # Calculate EU prediction: what axis value gives same EV as reference?
            ref_ev = ref.expected_value
            
            if r.axis == "Y":
                # Y-axis lottery EV = p_H * 100 + (1-p_H) * 50 = 50 + 50*p_H
                # Solve for p_H: ref_ev = 50 + 50*p_H => p_H = (ref_ev - 50) / 50
                eu_predicted = (ref_ev - 50) / 50
                eu_predicted = max(0, min(1, eu_predicted))  # Clamp to [0,1]
                indiff_pt = (0, r.indifference_value)
            else:
                # X-axis lottery EV = (1-p_L) * 50 + p_L * 0 = 50 - 50*p_L
                # Solve for p_L: ref_ev = 50 - 50*p_L => p_L = (50 - ref_ev) / 50
                eu_predicted = (50 - ref_ev) / 50
                eu_predicted = max(0, min(1, eu_predicted))  # Clamp to [0,1]
                indiff_pt = (r.indifference_value, 0)
            
            deviation = abs(r.indifference_value - eu_predicted)
            deviations.append(deviation)
            plot_data.append((ref, indiff_pt, deviation, r.axis))
        
        # Color by deviation: green = close to EU, red = far from EU
        if deviations:
            max_dev = max(deviations) if max(deviations) > 0 else 0.1
            norm = plt.Normalize(0, max_dev)
            cmap = plt.cm.RdYlGn_r  # Red = high deviation, green = low
            
            for ref, indiff_pt, dev, axis in plot_data:
                color = cmap(norm(dev))
                marker = 'o' if axis == "Y" else 's'
                
                ax.plot([ref.p_L, indiff_pt[0]], [ref.p_H, indiff_pt[1]], 
                       '-', color=color, linewidth=2, alpha=0.7)
                ax.scatter(ref.p_L, ref.p_H, c=[color], s=60, marker=marker, 
                          edgecolors='black', linewidth=0.5, zorder=5)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Deviation from EU Prediction')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$p_L$ (Probability of $0)', fontsize=12)
        ax.set_ylabel(r'$p_H$ (Probability of $100)', fontsize=12)
        ax.set_title('EU Deviation: Green = Rational, Red = Irrational', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        # Add mean deviation annotation
        mean_dev = np.mean(deviations) if deviations else 0
        ax.text(0.95, 0.95, f'Mean deviation: {mean_dev:.3f}', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    


    def save_results(self, output_dir: str = ".", analysis: Dict[str, Any] = None):
        """Save all results to files"""
        
        print("\nSaving results...")
        print("-"*70)
        
        # 1. CSV with main results
        csv_data = []
        for result in self.results:
            ref = result.reference_point
            csv_data.append({
                'p_L': ref.p_L,
                'p_H': ref.p_H,
                'p_M': ref.p_M,
                'expected_value': ref.expected_value,
                'axis': result.axis,
                'indifference_value': result.indifference_value,
                'n_iterations': result.n_iterations,
                'final_precision': result.final_precision,
                'timestamp': result.timestamp
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, "mm_triangle_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_path}")
        
        # 2. Detailed JSON with choice history
        json_data = []
        for result in self.results:
            ref = result.reference_point
            json_data.append({
                'reference_point': {
                    'p_L': ref.p_L,
                    'p_H': ref.p_H,
                    'p_M': ref.p_M,
                    'expected_value': ref.expected_value
                },
                'axis': result.axis,
                'indifference_value': result.indifference_value,
                'n_iterations': result.n_iterations,
                'final_precision': result.final_precision,
                'timestamp': result.timestamp,
                'choice_history': [
                    {
                        'iteration': c.iteration,
                        'midpoint': c.midpoint,
                        'choice': c.choice,
                        'response': c.raw_response
                    }
                    for c in result.choice_history
                ]
            })
        
        json_path = os.path.join(output_dir, "mm_triangle_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"  ✓ Saved: {json_path}")

        # 4. Save Copy for Web (web/data)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        web_data_dir = os.path.join(base_dir, "web", "data")
        os.makedirs(web_data_dir, exist_ok=True)
        
        # Use simple model name for filename
        global model_id
        if model_id:
            model_name_safe = model_id.replace("/", "_").replace(":", "_")
            web_json_path = os.path.join(web_data_dir, f"independence_results_{model_name_safe}.json")
            
            # Add analysis text if available
            web_data = list(json_data) # Shallow copy
            if analysis and isinstance(web_data, list):
                # We need to wrap it or attach it. Since JSON is a list of results, 
                # we can't easily add a top-level key without changing structure.
                # However, card.html expects a list for plotting.
                # Let's verify what card.html expects.
                # Actually, card.html just iterates the list.
                # We can wrap the list in a dict { "results": [...], "analysis_text": "..." }
                # BUT this breaks existing card.html logic which expects an array.
                # BETTER: Add it to the card.html logic? NO, we want to change data structure.
                # Let's change the output structure to include metadata.
                
                # Re-structuring output for WEB only to include metadata
                par = analysis.get('parallelism_test', {})
                text = "> DETAILS<br><br>"
                if par.get('eu_violated', False):
                    text += "Analysis indicates <b>SYSTEMATIC VIOLATIONS of Expected Utility Theory</b>."
                    pattern = par.get('fanning_pattern', 'Unknown')
                    text += f" The indifference curves exhibit a <b>{pattern}</b> pattern."
                    if 'FANNING_OUT' in pattern:
                        text += " This 'fanning out' behavior aligns with the Allais Paradox and typical human risk aversion."
                else:
                    text += "Analysis indicates the model adheres to <b>Expected Utility Theory</b>."
                    text += " Indifference curves remain parallel, satisfying the Independence Axiom."
                
                web_output = {
                    "results": json_data,
                    "analysis_text": text
                }
                
                with open(web_json_path, 'w') as f:
                    json.dump(web_output, f, indent=2)
            else:
                # Fallback to old format
                with open(web_json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
            print(f"  ✓ Saved for web: {web_json_path}")
            
            # Update models registry (models.json)
            registry_path = os.path.join(web_data_dir, "models.json")
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
        
        # 3. Validation results
        if self.validation_results:
            validation_data = []
            for orig, retest in self.validation_results:
                validation_data.append({
                    'p_L': orig.reference_point.p_L,
                    'p_H': orig.reference_point.p_H,
                    'original_value': orig.indifference_value,
                    'retest_value': retest.indifference_value,
                    'deviation': abs(orig.indifference_value - retest.indifference_value)
                })
            
            val_df = pd.DataFrame(validation_data)
            val_path = os.path.join(output_dir, "mm_triangle_validation.csv")
            val_df.to_csv(val_path, index=False)
            print(f"  ✓ Saved: {val_path}")
        
        print()
    
    def generate_report(self, analysis: Dict, output_dir: str = "."):
        """Generate comprehensive text report"""
        
        report_path = os.path.join(output_dir, "mm_triangle_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MARSCHAK-MACHINA TRIANGLE EXPERIMENT REPORT\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("EXPERIMENT SETUP\n")
            f.write("-"*70 + "\n")
            f.write(f"Outcomes: ${X_LOW} (Low), ${X_MID} (Middle), ${X_HIGH} (High)\n")
            f.write(f"Grid divisions: {self.n_divisions}\n")
            f.write(f"Bisection iterations: {self.n_iterations}\n")
            f.write(f"Precision: ~{1/2**self.n_iterations:.4f}\n")
            f.write(f"Points tested: {analysis['n_points_tested']}\n")
            f.write(f"Y-axis mappings: {analysis['n_y_axis_lotteries']}\n")
            f.write(f"X-axis mappings: {analysis['n_x_axis_lotteries']}\n\n")
            
            f.write("VALIDATION (CONSISTENCY CHECK)\n")
            f.write("-"*70 + "\n")
            val = analysis['validation']
            f.write(f"Retests performed: {val['n_retests']}\n")
            f.write(f"Mean deviation: {val['mean_deviation']:.4f}\n")
            f.write(f"Max deviation: {val['max_deviation']:.4f}\n")
            consistency = "GOOD" if val['mean_deviation'] < 0.05 else "MODERATE" if val['mean_deviation'] < 0.1 else "POOR"
            f.write(f"Consistency rating: {consistency}\n\n")
            
            # DIAGNOSTICS SECTION (NEW)
            if 'diagnostics' in analysis:
                f.write("DIAGNOSTICS\n")
                f.write("-"*70 + "\n")
                diag = analysis['diagnostics']
                
                f.write("Monotonicity (Dominance) Tests:\n")
                f.write(f"  Checks performed: {diag['monotonicity']['n_checks']}\n")
                f.write(f"  Passed: {diag['monotonicity']['n_passed']}\n")
                f.write(f"  Pass rate: {diag['monotonicity']['pass_rate']:.1%}\n\n")
                
                f.write("Transitivity Tests:\n")
                f.write(f"  Checks performed: {diag['transitivity']['n_checks']}\n")
                f.write(f"  Consistent: {diag['transitivity']['n_consistent']}\n")
                f.write(f"  Consistency rate: {diag['transitivity']['consistency_rate']:.1%}\n\n")
                
                f.write("Bidirectional Bisection (Positional Bias):\n")
                f.write(f"  Samples tested: {diag['bidirectional']['n_samples']}\n")
                f.write(f"  Mean difference: {diag['bidirectional']['mean_difference']:.4f}\n")
                f.write(f"  Positional bias: {'DETECTED' if diag['bidirectional']['positional_bias_detected'] else 'NOT DETECTED'}\n\n")
                
                if 'rationality_score' in diag:
                    f.write(f"Overall Rationality Score: {diag['rationality_score']:.2f} / 1.00\n\n")
            
            f.write("PARALLELISM TEST (INDEPENDENCE AXIOM)\n")
            f.write("-"*70 + "\n")
            par = analysis['parallelism_test']
            f.write(f"Mean slope: {par['mean_slope']:.4f}\n")
            f.write(f"Slope standard deviation: {par['slope_std']:.4f}\n")
            f.write(f"Coefficient of variation: {par.get('coefficient_of_variation', 0):.4f}\n")
            f.write(f"Slope-position correlation: {par.get('slope_position_correlation', 0):.4f}\n")
            f.write(f"Fanning pattern: {par['fanning_pattern']}\n")
            f.write(f"EU Violated: {'YES' if par['eu_violated'] else 'NO'}\n\n")
            
            f.write("QUADRATIC UTILITY FIT\n")
            f.write("-"*70 + "\n")
            quad = analysis['quadratic_fit']
            if quad.get('status') == 'SUCCESS':
                f.write("Linear coefficients (α):\n")
                f.write(f"  α_L (Low): {quad['alpha_L']:.6f}\n")
                f.write(f"  α_M (Mid): {quad['alpha_M']:.6f}\n")
                f.write(f"  α_H (High): {quad['alpha_H']:.6f}\n\n")
                f.write("Quadratic coefficients (β):\n")
                f.write(f"  β_LL: {quad['beta_LL']:.6f}\n")
                f.write(f"  β_MM: {quad['beta_MM']:.6f}\n")
                f.write(f"  β_HH: {quad['beta_HH']:.6f}\n")
                f.write(f"  β_LM: {quad['beta_LM']:.6f}\n")
                f.write(f"  β_LH: {quad['beta_LH']:.6f}\n")
                f.write(f"  β_MH: {quad['beta_MH']:.6f}\n\n")
                f.write(f"||β|| (norm): {quad['beta_norm']:.6f}\n")
                f.write(f"Is EU? (||β|| < 0.05): {quad['is_eu']}\n")
                f.write(f"Interpretation: {quad['interpretation']}\n")
                f.write(f"Residual loss: {quad['residual_loss']:.6f}\n")
            else:
                f.write(f"Status: {quad.get('status', 'ERROR')}\n")
            f.write("\n")
            
            f.write("OVERALL CONCLUSION\n")
            f.write("-"*70 + "\n")
            if not par['eu_violated']:
                f.write("✓ The LLM appears to be an EXPECTED UTILITY MAXIMIZER.\n")
                f.write("  Indifference curves are approximately parallel.\n")
                f.write("  The Independence Axiom holds for this model's preferences.\n")
            else:
                f.write("✗ The LLM exhibits EXPECTED UTILITY VIOLATIONS.\n")
                f.write(f"  Pattern detected: {par['fanning_pattern']}\n")
                if 'FANNING_OUT' in par['fanning_pattern']:
                    f.write("  This is consistent with the Allais Paradox pattern.\n")
                    f.write("  The model may overweight certainty or display probability weighting.\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"  ✓ Saved: {report_path}")


# -------------------------------------------------------------
# 9. Main Execution
# -------------------------------------------------------------

def main():
    """Main execution function"""
    global model_id, llm, PRINT_INTERACTIONS

    parser = argparse.ArgumentParser(description="MM Triangle Experiment")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model ID to use")
    parser.add_argument("--no-print", action="store_true", help="Disable printing interactions")
    parser.add_argument("--divisions", type=int, default=12, help="Number of grid divisions")
    parser.add_argument("--iterations", type=int, default=10, help="Number of bisection iterations")
    args = parser.parse_args()

    model_id = args.model
    PRINT_INTERACTIONS = not args.no_print

    print(f"Loading model interface for: {model_id}")
    try:
        llm = get_model_interface(model_id)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure the model is defined in src/models/registry.py")
        sys.exit(1)
    
    # Create output directory with model name
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Sanitize model_id for directory name (replace / with _)
    model_name_safe = model_id.replace("/", "_").replace(":", "_")
    
    # Use standard results directory structure: data/results/independence/{model_id}
    output_subdir = os.path.join(base_dir, "data", "results", "independence", model_name_safe)
    os.makedirs(output_subdir, exist_ok=True)
    
    print(f"\nOutput directory: {output_subdir}\n")
    
    # Initialize experiment
    # Using n_divisions=7 gives approximately 36 interior grid points
    # Each point requires 10 bisection iterations = 360 total LLM calls
    # Plus validation (~36 more calls)
    experiment = MMTriangleExperiment(
        n_divisions=args.divisions,
        n_iterations=args.iterations,
        validation_fraction=0.1,
        print_progress=True
    )
    
    # Run full experiment
    analysis = experiment.run_full_experiment()
    
    # Generate visualizations
    experiment.generate_visualizations(output_dir=output_subdir)
    
    # Save all results
    experiment.save_results(output_dir=output_subdir, analysis=analysis)
    
    # Generate final report
    experiment.generate_report(analysis, output_dir=output_subdir)
    
    print("\n" + "="*70)
    print(" "*20 + "EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  📄 mm_triangle_report.txt      - Comprehensive analysis report")
    print(f"  📊 mm_triangle_results.csv     - Main results table")
    print(f"  📋 mm_triangle_results.json    - Detailed results with choice history")
    print(f"  📊 mm_triangle_validation.csv  - Consistency check results")
    print(f"  📈 mm_triangle_grid.png        - Triangle with tested points")
    print(f"  📈 indifference_curves.png     - Traced indifference curves")
    print(f"  📈 slope_heatmap.png           - Local slope heat map")
    print(f"  📈 slope_vector_field.png      - Direction field visualization")
    print(f"  📈 eu_comparison.png           - EU prediction vs actual")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
