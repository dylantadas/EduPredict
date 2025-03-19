import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

def perform_automated_eda(datasets):
    """Performs automated eda and displays results identifying potential data quality issues."""

    for name, df in datasets.items():
        print(f"\n{'='*50}")
        print(f"Analysis of {name}")
        print(f"{'='*50}")

        # dataset overview
        print("\nDataset Overview:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumns and their data types:")
        print(df.dtypes)

        # missing values
        print("\nMissing Values Analysis:")
        missing = df.isnull().sum()
        if missing.any():
            print(missing[missing > 0])
        else:
            print("No missing values found")

        # numerical analysis
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            print("\nNumerical Columns Summary:")
            print(df[numerical_cols].describe())

        # categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\nCategorical Columns Summary:")
            for col in categorical_cols:
                print(f"\nDistribution of {col}:")
                print(df[col].value_counts(normalize=True).head())


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


def run_eda_pipeline(datasets):
    """Executes complete eda pipeline, displaying insights about data quality, student performance, and engagement patterns."""

    print("\nStarting EDA...")
    perform_automated_eda(datasets)
    analyze_student_performance(datasets)
    analyze_engagement_patterns(datasets)

    print("\nEDA Complete.")


def visualize_demographic_distributions(student_info: pd.DataFrame, save_path: Optional[str] = None):
    """Visualizes distributions of key demographic variables."""
    
    demo_cols = ['gender', 'age_band', 'imd_band', 'region', 'highest_education', 'disability']
    available_cols = [col for col in demo_cols if col in student_info.columns]
    
    # calculate number of plots needed
    n_cols = min(2, len(available_cols))
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    
    # create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # create plots
    for i, col in enumerate(available_cols):
        if i < len(axes):
            value_counts = student_info[col].value_counts().sort_values(ascending=False)
            ax = axes[i]
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
            # add percentage labels
            total = value_counts.sum()
            for j, p in enumerate(ax.patches):
                percentage = f'{100 * p.get_height() / total:.1f}%'
                ax.annotate(percentage, 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='bottom')
    
    # hide unused subplots
    for i in range(len(available_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved demographic visualizations to {save_path}")
    
    plt.show()


def visualize_performance_by_demographics(student_data: pd.DataFrame, 
                                         demo_cols: List[str] = ['gender', 'age_band', 'imd_band'],
                                         save_path: Optional[str] = None):
    """Visualizes student performance across demographic groups."""
    
    available_cols = [col for col in demo_cols if col in student_data.columns]
    
    if 'final_result' not in student_data.columns:
        print("Error: final_result column not found in data")
        return
    
    # calculate number of plots needed
    n_cols = min(2, len(available_cols))
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    
    # create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # create plots
    for i, col in enumerate(available_cols):
        if i < len(axes):
            # calculate pass rates by demographic group
            pass_rate = (
                student_data
                .groupby(col)
                .apply(lambda x: (x['final_result'].str.lower().isin(['pass', 'distinction'])).mean())
                .sort_values()
            )
            
            ax = axes[i]
            pass_rate.plot(kind='barh', ax=ax)
            ax.set_title(f'Pass Rate by {col}')
            ax.set_xlabel('Pass Rate')
            ax.set_xlim(0, 1)
            
            # add percentage labels
            for j, p in enumerate(ax.patches):
                percentage = f'{100 * p.get_width():.1f}%'
                ax.annotate(percentage, 
                           (p.get_width(), p.get_y() + p.get_height() / 2.), 
                           ha='left', va='center')
    
    # hide unused subplots
    for i in range(len(available_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance visualizations to {save_path}")
    
    plt.show()


def visualize_engagement_patterns(vle_data: pd.DataFrame, 
                                final_results: Optional[pd.DataFrame] = None,
                                save_path: Optional[str] = None):
    """Visualizes temporal engagement patterns, optionally split by final result."""
    
    # weekly aggregation
    vle_data['week'] = vle_data['date'] // 7
    weekly_data = vle_data.groupby('week')['sum_click'].agg(['mean', 'count', 'sum'])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # plot total clicks over time
    weekly_data['sum'].plot(ax=axes[0])
    axes[0].set_title('Total Clicks by Week')
    axes[0].set_xlabel('Week')
    axes[0].set_ylabel('Total Clicks')
    axes[0].grid(True)
    
    # plot average clicks per student
    weekly_data['mean'].plot(ax=axes[1])
    axes[1].set_title('Average Clicks per Interaction by Week')
    axes[1].set_xlabel('Week')
    axes[1].set_ylabel('Average Clicks')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # create second figure for engagement by final result
    if final_results is not None and 'id_student' in vle_data.columns and 'final_result' in final_results.columns:
        # merge data
        merged_data = vle_data.merge(
            final_results[['id_student', 'final_result']],
            on='id_student',
            how='left'
        )
        
        # aggregate by week and final result
        result_weekly = merged_data.groupby(['week', 'final_result'])['sum_click'].mean().unstack()
        
        plt.figure(figsize=(14, 6))
        result_weekly.plot()
        plt.title('Average Clicks by Week and Final Result')
        plt.xlabel('Week')
        plt.ylabel('Average Clicks')
        plt.grid(True)
        plt.legend(title='Final Result')
        plt.tight_layout()
    
    # save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved engagement visualizations to {save_path}")
    
    plt.show()