import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import glob

plt.rcParams["figure.constrained_layout.use"] = True


class ResultAggregator:
    """Class to aggregate and visualize results from evolutionary algorithm experiments."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.brain_variants = ["nobrain", "selfadaptive", "uniform"]
        self.data: Dict[str, List[pd.DataFrame]] = {}
        
    def load_data(self) -> None:
        """Load all CSV files for each brain variant."""
        for variant in self.brain_variants:
            self.data[variant] = []
            for run in range(1, 4):  # runs 1, 2, 3
                csv_file = os.path.join(self.results_dir, f"fitness_progression_{variant}_{run}.csv")
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    self.data[variant].append(df)
                    print(f"Loaded {csv_file}")
                else:
                    print(f"Warning: {csv_file} not found")
    
    def plot_brain_variant(self, variant: str, save_plot: bool = True) -> None:
        """Create a plot for a specific brain variant showing average across all runs."""
        if variant not in self.data or not self.data[variant]:
            print(f"No data available for {variant}")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Align all dataframes by generation (in case they have different lengths)
        min_gens = min(len(df) for df in self.data[variant])
        aligned_means = np.array([df['mean'].iloc[:min_gens].values for df in self.data[variant]])
        generations = self.data[variant][0]['gen'].iloc[:min_gens].values
        
        # Calculate overall statistics across all runs for main variant
        overall_mean = np.mean(aligned_means, axis=0)
        overall_std = np.std(aligned_means, axis=0)  # Standard deviation across runs
        
        # Plot main variant
        variant_color = 'blue'
        variant_label = variant.replace("_", " ").title()
        
        plt.plot(generations, overall_mean, 
                color=variant_color, linewidth=2,
                label=f'{variant_label} - Moving Average')
        
        plt.fill_between(generations, 
                       overall_mean - overall_std, 
                       overall_mean + overall_std,
                       color=variant_color, alpha=0.3,
                       label=f'{variant_label} - ± Moving Std Dev')
        
        # Add baseline (nobrain) for comparison if this is selfadaptive or uniform
        if variant in ['selfadaptive', 'uniform'] and 'nobrain' in self.data and self.data['nobrain']:
            # Calculate baseline statistics
            baseline_min_gens = min(len(df) for df in self.data['nobrain'])
            baseline_aligned_means = np.array([df['mean'].iloc[:baseline_min_gens].values for df in self.data['nobrain']])
            baseline_generations = self.data['nobrain'][0]['gen'].iloc[:baseline_min_gens].values
            
            baseline_mean = np.mean(baseline_aligned_means, axis=0)
            baseline_std = np.std(baseline_aligned_means, axis=0)
            
            # Align baseline data with current variant data
            common_gens = min(len(baseline_generations), len(generations))
            
            plt.plot(baseline_generations[:common_gens], baseline_mean[:common_gens], 
                    color='red', linewidth=2, linestyle='--',
                    label='Nobrain (Baseline) - Moving Average')
            
            plt.fill_between(baseline_generations[:common_gens], 
                           baseline_mean[:common_gens] - baseline_std[:common_gens], 
                           baseline_mean[:common_gens] + baseline_std[:common_gens],
                           color='red', alpha=0.2,
                           label='Nobrain (Baseline) - ± Moving Std Dev')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'Fitness Evolution - {variant.replace("_", " ").title()} Brain vs Baseline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0,0.9)
        
        if save_plot:
            plt.savefig(f"{self.results_dir}/aggregated_{variant}_fitness.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Saved plot: aggregated_{variant}_fitness.png")
        
        plt.show()
    
    
    def print_summary_statistics(self) -> None:
        """Print summary statistics for all brain variants."""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        for variant in self.brain_variants:
            if variant not in self.data or not self.data[variant]:
                continue
                
            print(f"\n{variant.replace('_', ' ').title()} Brain:")
            print("-" * 40)
            
            # Calculate statistics across all runs
            final_fitnesses = []
            max_fitnesses = []
            
            for i, df in enumerate(self.data[variant]):
                final_fitness = df['mean'].iloc[-1]
                max_fitness = df['highest'].max()
                final_fitnesses.append(final_fitness)
                max_fitnesses.append(max_fitness)
                
                print(f"  Run {i+1}: Final fitness = {final_fitness:.4f}, "
                      f"Max fitness = {max_fitness:.4f}")
            
            if final_fitnesses:
                print(f"  Average final fitness: {np.mean(final_fitnesses):.4f} ± {np.std(final_fitnesses):.4f}")
                print(f"  Average max fitness: {np.mean(max_fitnesses):.4f} ± {np.std(max_fitnesses):.4f}")
    
    def generate_all_plots(self) -> None:
        """Generate all plots and summary statistics."""
        print("Loading data...")
        self.load_data()
        
        print("\nGenerating individual brain variant plots...")
        for variant in self.brain_variants:
            self.plot_brain_variant(variant)
        
        print("\nGenerating summary statistics...")
        self.print_summary_statistics()
        
        print(f"\nAll plots saved to {self.results_dir}/ directory")


if __name__ == "__main__":
    # Initialize the aggregator
    aggregator = ResultAggregator("results")
    
    # Generate all plots and statistics
    aggregator.generate_all_plots()
