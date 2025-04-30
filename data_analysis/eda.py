import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from visualization.visualization_runner import VisualizationRunner


def analyze_student_performance(datasets):
    """Analyzes student performance patterns, displays demographic-related assessment score insights."""

    student_data = datasets['student_info'].merge(
        datasets['student_assessments'],
        on='id_student'
    )

    print("\nStudent Performance Analysis")
    print("="*50)

    # overall score distribution
    print("\nScore Distribution:")
    print(student_data['score'].describe())

    # performance by demographic groups
    for column in ['gender', 'age_band', 'imd_band']:
        print(f"\nAverage Score by {column}:")
        print(student_data.groupby(column, observed=False)['score'].mean())


def analyze_engagement_patterns(datasets):
    """Examines student engagement in vle, displays patterns in interaction frequency and timing."""

    vle_data = datasets['vle_interactions']

    print("\nEngagement Pattern Analysis")
    print("="*50)

    # overall engagement metrics
    print("\nDaily Interaction Summary:")
    daily_interactions = vle_data.groupby('id_student')['sum_click'].sum()
    print(daily_interactions.describe())

    # activity timing
    print("\nTemporal Distribution of Activities:")
    activity_timing = vle_data.groupby('date')['sum_click'].mean()
    print(activity_timing.describe())


def document_eda_findings(datasets):
    """Documents key findings from eda to inform modeling decisions."""
    
    findings = {
        "demographic_insights": [],
        "performance_patterns": [],
        "engagement_patterns": [],
        "data_quality_issues": [],
        "modeling_implications": []
    }
    
    # demographic insights
    student_data = datasets['student_info']
    findings["demographic_insights"].append(
        f"Age distribution: {student_data['age_band'].value_counts(normalize=True).to_dict()}"
    )
    findings["demographic_insights"].append(
        f"Gender distribution: {student_data['gender'].value_counts(normalize=True).to_dict()}"
    )
    findings["demographic_insights"].append(
        f"IMD band distribution: {student_data['imd_band'].value_counts(normalize=True).to_dict()}"
    )
    
    # performance patterns
    if 'student_assessments' in datasets and 'student_info' in datasets:
        performance_data = datasets['student_info'].merge(
            datasets['student_assessments'],
            on='id_student'
        )
        
        # age-performance relationship
        age_perf = performance_data.groupby('age_band', observed=False)['score'].mean()
        findings["performance_patterns"].append(f"Age-performance relationship: {age_perf.to_dict()}")
        
        # Check if age bands show significant score differences
        age_bands = age_perf.index.tolist()
        if len(age_bands) >= 2 and max(age_perf) - min(age_perf) > 5:
            # Only add this implication if there's at least a 5-point difference in scores
            findings["modeling_implications"].append(
                f"Age shows correlation with performance (difference of {max(age_perf) - min(age_perf):.1f} points); "
                f"age-specific features may be beneficial"
            )
        
        # imd-performance relationship
        imd_perf = performance_data.groupby('imd_band', observed=False)['score'].mean()
        findings["performance_patterns"].append(f"IMD-performance relationship: {imd_perf.to_dict()}")
        
        # Check if IMD bands show socioeconomic gradient
        # Get numeric IMD bands (e.g., 0-10%, 10-20%, etc.)
        numeric_imd_bands = [band for band in imd_perf.index if '-' in str(band) and '%' in str(band)]
        if len(numeric_imd_bands) >= 3:
            # Sort by socioeconomic status (lower percentile = more deprived)
            sorted_bands = sorted(numeric_imd_bands, key=lambda x: int(str(x).split('-')[0].replace('%', '')))
            sorted_scores = [imd_perf[band] for band in sorted_bands]
            
            # Check if there's a consistent trend (increasing or decreasing)
            from scipy import stats
            if len(sorted_scores) >= 3:
                # Compute correlation between rank and score
                correlation, p_value = stats.spearmanr([sorted_bands.index(band) for band in sorted_bands], sorted_scores)
                if abs(correlation) > 0.5 and p_value < 0.05:
                    findings["modeling_implications"].append(
                        f"IMD band shows socioeconomic gradient in performance (correlation={correlation:.2f}, p={p_value:.3f}); "
                        f"fairness metrics needed"
                    )
    
    # engagement patterns
    if 'vle_interactions' in datasets:
        vle_data = datasets['vle_interactions']
        
        # activity distribution over time
        activity_time = vle_data.groupby('date')['sum_click'].mean()
        findings["engagement_patterns"].append(
            f"Peak engagement day: Day {activity_time.idxmax()} with {activity_time.max():.2f} avg clicks"
        )
        
        # student engagement variability
        student_engagement = vle_data.groupby('id_student')['sum_click'].sum()
        cv = student_engagement.std() / student_engagement.mean() if student_engagement.mean() > 0 else 0
        findings["engagement_patterns"].append(
            f"Engagement variability: min={student_engagement.min()}, max={student_engagement.max()}, " 
            f"median={student_engagement.median()}, std={student_engagement.std():.2f}, CV={cv:.2f}"
        )
        
        # Add implication only if coefficient of variation (CV) is high (>1.0)
        if cv > 1.0:
            findings["modeling_implications"].append(
                f"High variability in engagement (CV={cv:.2f}); temporal features will be critical"
            )
    
    # data quality issues
    for name, df in datasets.items():
        missing = df.isnull().sum()
        if missing.any():
            findings["data_quality_issues"].append(
                f"{name} has missing values: {missing[missing > 0].to_dict()}"
            )
            
    # Add implication about missing data if significant
    missing_counts = {name: df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) 
                     for name, df in datasets.items()}
    if any(pct > 0.01 for pct in missing_counts.values()):
        findings["modeling_implications"].append(
            f"Data contains missing values (up to {max(missing_counts.values())*100:.1f}% in some datasets); "
            f"robust imputation strategies needed"
        )
    
    return findings


def perform_automated_eda(clean_data: Dict[str, pd.DataFrame], 
                         viz_dir: str,
                         report_dir: str):
    """Perform automated exploratory data analysis."""
    
    # Map clean_data keys to expected dataset keys
    datasets = {
        'student_info': clean_data['demographics'],
        'vle_interactions': clean_data['vle'],
        'student_assessments': clean_data['assessments']
    }
    
    # Get documented findings
    eda_findings = document_eda_findings(datasets)
    
    # Save findings
    with open(os.path.join(report_dir, 'eda_findings.json'), 'w') as f:
        json.dump(eda_findings, f, indent=2)
    
    # Use visualization runner for all visualizations
    viz_runner = VisualizationRunner(viz_dir)
    
    # Run all relevant visualizations with correct keys
    demo_paths = viz_runner.run_demographic_visualizations(clean_data['demographics'])
    perf_paths = viz_runner.run_performance_visualizations(
        clean_data,
        {'demographics': True}
    )
    engagement_paths = viz_runner.run_engagement_visualizations(
        clean_data['vle'],
        clean_data['demographics']
    )
    
    return {
        'findings': eda_findings,
        'visualization_paths': {
            'demographics': demo_paths,
            'performance': perf_paths,
            'engagement': engagement_paths
        }
    }