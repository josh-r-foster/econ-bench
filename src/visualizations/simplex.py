import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from typing import List, Tuple, Any

class SimplexVisualizer:
    """Visualization tools for Marschak-Machina triangle data"""
    
    def __init__(self, outcomes: List[float]):
        self.outcomes = outcomes

    @staticmethod
    def simplex_to_cartesian(p_high: float, p_mid: float, p_low: float) -> Tuple[float, float]:
        """
        Convert to Marschak-Machina (Right-Angled) Coordinates
        X-axis: Probability of Worst Outcome ($0) -> p_low
        Y-axis: Probability of Best Outcome (highest $) -> p_high
        """
        x = p_low
        y = p_high
        return x, y
    
    def plot_utility_surface(self, points: np.ndarray, utilities: np.ndarray,
                           save_path: str, title: str = "Utility Surface"):
        """Plot the utility surface on a Marschak-Machina triangle"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        x_coords = []
        y_coords = []
        for pt in points:
            x, y = self.simplex_to_cartesian(pt[0], pt[1], pt[2])
            x_coords.append(x)
            y_coords.append(y)
        
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        
        # Take log for better visualization
        log_utilities = np.log(utilities + 1e-10)
        
        triang = mtri.Triangulation(x_coords, y_coords)
        
        levels = np.linspace(log_utilities.min(), log_utilities.max(), 20)
        tcf = ax.tricontourf(triang, log_utilities, levels=levels, cmap='viridis')
        plt.colorbar(tcf, ax=ax, label='ln(Utility)')
        
        ax.tricontour(triang, log_utilities, levels=levels, colors='white', 
                     linewidths=0.5, alpha=0.5)
        
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        ax.text(0, 1.05, f'100% ${self.outcomes[2]}\\n(Best)', ha='center', fontsize=10, fontweight='bold')
        ax.text(1.05, 0, f'100% ${self.outcomes[0]}\\n(Worst)', ha='left', va='center', fontsize=10, fontweight='bold')
        ax.text(-0.1, -0.05, f'100% ${self.outcomes[1]}\\n(Middle)', ha='right', fontsize=10, fontweight='bold')
        
        ax.set_xlabel(f'Probability of Worst Outcome (${self.outcomes[0]})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Probability of Best Outcome (${self.outcomes[2]})', fontsize=12, fontweight='bold')
        
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_chain_trajectory(self, chain_states: List[Any], save_path: str,
                             title: str = "MCMC Chain Trajectory"):
        """Plot the trajectory of the MCMC chain"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        coords = []
        for s in chain_states:
            g = s.gamble
            x, y = self.simplex_to_cartesian(g.p_high, g.p_mid, g.p_low)
            coords.append((x, y))
        
        coords = np.array(coords)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(coords)))
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=10, alpha=0.5)
        
        if len(coords) > 0:
            ax.scatter([coords[0, 0]], [coords[0, 1]], c='green', s=200, 
                      marker='*', label='Start', zorder=10)
            ax.scatter([coords[-1, 0]], [coords[-1, 1]], c='red', s=200, 
                      marker='*', label='End', zorder=10)
        
        ax.text(0, 1.05, f'100% ${self.outcomes[2]}', ha='center', fontsize=10)
        ax.text(1.05, 0, f'100% ${self.outcomes[0]}', ha='left', va='center', fontsize=10)
        ax.text(-0.1, -0.05, f'100% ${self.outcomes[1]}', ha='right', fontsize=10)
        
        ax.set_xlabel(f'Probability of ${self.outcomes[0]}', fontsize=12)
        ax.set_ylabel(f'Probability of ${self.outcomes[2]}', fontsize=12)
        
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_density_histogram(self, points: np.ndarray, save_path: str):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        x_coords = []
        y_coords = []
        for pt in points:
            x, y = self.simplex_to_cartesian(pt[0], pt[1], pt[2])
            x_coords.append(x)
            y_coords.append(y)
        
        ax.hist2d(x_coords, y_coords, bins=30, cmap='Blues')
        
        triangle = plt.Polygon([(0, 0), (1, 0), (0, 1)], 
                               fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(triangle)
        
        ax.set_xlabel(f'Probability of ${self.outcomes[0]} (Worst)', fontsize=12)
        ax.set_ylabel(f'Probability of ${self.outcomes[2]} (Best)', fontsize=12)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.set_aspect('equal')
        ax.set_title('MCMC Sample Density', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
