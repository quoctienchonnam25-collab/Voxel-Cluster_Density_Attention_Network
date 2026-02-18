#!/usr/bin/env python3
"""
Visualize and analyze AD brain age gap prediction results.

This script generates comprehensive visualizations and statistical analyses
of the brain age gap predictions for AD/MCI patients.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json
import argparse


def load_results(predictions_file: Path) -> pd.DataFrame:
    """Load prediction results."""
    print(f"Loading results from {predictions_file}...")
    df = pd.read_csv(predictions_file)
    print(f"  ✓ Loaded {len(df)} samples")
    return df


def plot_brain_age_gap_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    cohort_column: str = 'diagnosis'
):
    """Plot brain age gap distribution by cohort."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram
    ax = axes[0, 0]
    for cohort in df[cohort_column].unique():
        cohort_data = df[df[cohort_column] == cohort]['ensemble_corrected_delta']
        ax.hist(cohort_data, alpha=0.6, label=cohort, bins=30)
    ax.set_xlabel('Brain Age Gap (years)')
    ax.set_ylabel('Frequency')
    ax.set_title('Brain Age Gap Distribution by Cohort')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Box plot
    ax = axes[0, 1]
    cohorts = df[cohort_column].unique()
    data = [df[df[cohort_column] == c]['ensemble_corrected_delta'] for c in cohorts]
    bp = ax.boxplot(data, labels=cohorts, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette('Set2', len(cohorts))):
        patch.set_facecolor(color)
    ax.set_ylabel('Brain Age Gap (years)')
    ax.set_title('Brain Age Gap by Cohort')
    ax.grid(alpha=0.3)
    
    # 3. Violin plot
    ax = axes[1, 0]
    sns.violinplot(data=df, x=cohort_column, y='ensemble_corrected_delta', ax=ax)
    ax.set_ylabel('Brain Age Gap (years)')
    ax.set_title('Brain Age Gap Distribution (Violin Plot)')
    ax.grid(alpha=0.3)
    
    # 4. Scatter: True Age vs Brain Age Gap
    ax = axes[1, 1]
    for cohort in df[cohort_column].unique():
        cohort_data = df[df[cohort_column] == cohort]
        ax.scatter(
            cohort_data['true_age'],
            cohort_data['ensemble_corrected_delta'],
            alpha=0.6,
            label=cohort,
            s=50
        )
    ax.set_xlabel('Chronological Age (years)')
    ax.set_ylabel('Brain Age Gap (years)')
    ax.set_title('Brain Age Gap vs Chronological Age')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'brain_age_gap_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_regional_analysis(df: pd.DataFrame, output_dir: Path, num_regions: int = 32):
    """Plot regional brain age gap analysis."""
    
    # Extract regional delta-age columns
    regional_cols = [f'region_{i:02d}_delta_corrected' for i in range(num_regions)]
    regional_data = df[regional_cols]
    
    # Calculate mean delta-age per region
    mean_delta = regional_data.mean()
    std_delta = regional_data.std()
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Bar plot of mean delta-age per region
    ax = axes[0]
    regions = range(num_regions)
    colors = ['red' if x > 0 else 'blue' for x in mean_delta]
    ax.bar(regions, mean_delta, color=colors, alpha=0.7)
    ax.errorbar(regions, mean_delta, yerr=std_delta, fmt='none', ecolor='black', alpha=0.5)
    ax.set_xlabel('Brain Region')
    ax.set_ylabel('Mean Brain Age Gap (years)')
    ax.set_title('Regional Brain Age Gap Analysis')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3)
    
    # 2. Heatmap of regional correlations
    ax = axes[1]
    corr_matrix = regional_data.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax, 
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Regional Brain Age Gap Correlations')
    
    plt.tight_layout()
    output_file = output_dir / 'regional_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def calculate_cohort_statistics(
    df: pd.DataFrame,
    cohort_column: str = 'diagnosis'
) -> pd.DataFrame:
    """Calculate statistics for each cohort."""
    
    stats_list = []
    
    for cohort in df[cohort_column].unique():
        cohort_data = df[df[cohort_column] == cohort]['ensemble_corrected_delta']
        
        # Calculate statistics
        n = len(cohort_data)
        mean = cohort_data.mean()
        std = cohort_data.std()
        median = cohort_data.median()
        q25 = cohort_data.quantile(0.25)
        q75 = cohort_data.quantile(0.75)
        
        # One-sample t-test (H0: mean = 0)
        t_stat, p_value = stats.ttest_1samp(cohort_data, 0)
        
        # Cohen's d
        cohens_d = mean / std
        
        # 95% CI
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=stats.sem(cohort_data))
        
        stats_list.append({
            'Cohort': cohort,
            'N': n,
            'Mean': mean,
            'SD': std,
            'Median': median,
            'Q25': q25,
            'Q75': q75,
            'CI_lower': ci[0],
            'CI_upper': ci[1],
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d
        })
    
    return pd.DataFrame(stats_list)


def perform_between_group_tests(
    df: pd.DataFrame,
    cohort_column: str = 'diagnosis'
) -> pd.DataFrame:
    """Perform statistical tests between cohorts."""
    
    cohorts = df[cohort_column].unique()
    comparisons = []
    
    for i, cohort1 in enumerate(cohorts):
        for cohort2 in cohorts[i+1:]:
            data1 = df[df[cohort_column] == cohort1]['ensemble_corrected_delta']
            data2 = df[df[cohort_column] == cohort2]['ensemble_corrected_delta']
            
            # Independent t-test
            t_stat, p_ttest = stats.ttest_ind(data1, data2)
            
            # Welch's t-test (unequal variances)
            t_welch, p_welch = stats.ttest_ind(data1, data2, equal_var=False)
            
            # Mann-Whitney U test
            u_stat, p_mann = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
            cohens_d = (data1.mean() - data2.mean()) / pooled_std
            
            comparisons.append({
                'Comparison': f'{cohort1} vs {cohort2}',
                'Mean_diff': data1.mean() - data2.mean(),
                't_stat': t_stat,
                'p_ttest': p_ttest,
                'p_welch': p_welch,
                'p_mann_whitney': p_mann,
                'cohens_d': cohens_d
            })
    
    return pd.DataFrame(comparisons)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Visualize AD brain age gap prediction results'
    )
    parser.add_argument(
        '--predictions_file',
        type=str,
        required=True,
        help='Path to predictions CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--cohort_column',
        type=str,
        default='diagnosis',
        help='Column name for cohort/diagnosis'
    )
    parser.add_argument(
        '--num_regions',
        type=int,
        default=32,
        help='Number of brain regions'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AD BRAIN AGE GAP - VISUALIZATION & ANALYSIS")
    print("=" * 80)
    
    # Load results
    df = load_results(Path(args.predictions_file))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_brain_age_gap_distribution(df, output_dir, args.cohort_column)
    plot_regional_analysis(df, output_dir, args.num_regions)
    
    # Calculate statistics
    print("\nCalculating statistics...")
    cohort_stats = calculate_cohort_statistics(df, args.cohort_column)
    between_group_stats = perform_between_group_tests(df, args.cohort_column)
    
    # Save statistics
    cohort_stats.to_csv(output_dir / 'cohort_statistics.csv', index=False)
    between_group_stats.to_csv(output_dir / 'between_group_comparisons.csv', index=False)
    
    print(f"  ✓ Saved: cohort_statistics.csv")
    print(f"  ✓ Saved: between_group_comparisons.csv")
    
    # Print summary
    print("\n" + "=" * 80)
    print("COHORT STATISTICS")
    print("=" * 80)
    print(cohort_stats.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("BETWEEN-GROUP COMPARISONS")
    print("=" * 80)
    print(between_group_stats.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
