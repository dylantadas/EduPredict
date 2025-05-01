# Data Preprocessing Rationale

This document explains the rationale behind key data preprocessing decisions in EduPredict 2.0, particularly focusing on handling imbalanced features and demographic variables.

## Standardization of Protected Attributes

### Gender
- Values standardized to lowercase ('m', 'f') for consistency
- Rationale: Eliminates case-sensitivity issues while maintaining interpretability

### Region
- Values standardized to lowercase
- Rationale: Ensures consistent matching and comparison across the dataset

### Age Band
- Merged '35-55' and '55<=' into '35+'
- Rationale: 
  - The '55<=' group was severely underrepresented (only 216 samples)
  - Merging reduces imbalance while maintaining meaningful age distinction
  - Binary split (0-35 vs 35+) aligns with typical adult education patterns

### IMD Band (Index of Multiple Deprivation)
- Standardized format with '%' suffix
- Maintained all original bands
- Rationale:
  - Consistent formatting improves readability and prevents parsing errors
  - Preserved granular socioeconomic information due to its importance in educational outcomes

## Feature Transformations

### num_of_prev_attempts
- Applied within-group normalization for each protected attribute
- Rationale:
  - Raw values showed significant disparities across demographic groups
  - Normalization within groups helps prevent bias amplification
  - Maintains relative differences within groups while equalizing scales across groups

## Handling Imbalanced Features

### Demographic Imbalances
- Gender: Moderate imbalance (M: 54.8%, F: 45.2%) - Maintained as is
  - Rationale: Reflects actual student population distribution
  - Imbalance not severe enough to require correction

- Region: Varying representation (1.8k-3.4k per region) - Maintained as is
  - Rationale: Reflects actual geographic distribution
  - Used in stratification during data splitting to maintain proportions

- Age Band: Significant imbalance addressed through merging
  - Before: 0-35 (70.4%), 35-55 (28.9%), 55<= (0.7%)
  - After: 0-35 (70.4%), 35+ (29.6%)
  - Rationale: Improves model stability while maintaining key age-related patterns

### Feature-Level Imbalances
- num_of_prev_attempts: Normalized within demographic groups
  - Rationale: Preserves relative patterns while reducing demographic bias

## Data Quality Monitoring

- Implemented validation checks for:
  - Case consistency in categorical variables
  - Format consistency in IMD bands
  - Valid value ranges for numerical features
  - Group size monitoring for protected attributes

## Recent Fairness Enhancements (May 2025)

### Fairness-Aware Missing Value Imputation

The handling of missing values in `num_of_prev_attempts` has been enhanced with a fairness-aware imputation strategy that:

1. Maintains demographic balance by:
   - Computing group-specific statistics (mean, std, count) for each protected attribute
   - Using weighted sampling based on group sizes to prevent majority group bias
   - Generating imputed values from group-specific distributions

2. Applies fairness constraints through:
   - Group-specific normalization to maintain relative differences
   - Weighted adjustments based on group representation
   - Non-negativity constraints to maintain data validity

3. Validates fairness metrics by:
   - Monitoring disparity changes after imputation
   - Checking against configured fairness thresholds
   - Warning when disparities exceed acceptable levels

### Bias Mitigation Strategies

The preprocessing pipeline now includes several bias mitigation techniques:

1. Protected Attribute Handling:
   - Standardization of values using PROTECTED_ATTRIBUTES config
   - Proportional distribution of invalid values
   - Preservation of group distributions during cleaning

2. Sampling Strategies:
   - Stratified sampling for row filtering
   - Group-balanced oversampling/undersampling
   - Distribution-preserving value imputation

3. Fairness Monitoring:
   - Continuous tracking of demographic disparities
   - Group-specific distribution validation
   - Impact assessment of data transformations

### Latest Enhancements (May 2025)

#### Advanced Fairness-Aware Imputation for num_of_prev_attempts

The imputation strategy for num_of_prev_attempts has been further refined to ensure complete coverage and fairness:

1. Group-Specific Statistics:
   - Calculates detailed statistics (mean, std, min, max) for each demographic group
   - Uses robust fallback to global statistics when group data is insufficient
   - Applies group-specific scaling to maintain relative patterns

2. Weighted Imputation:
   - Employs fairness weights based on group representation
   - Ensures balanced contribution from all demographic groups
   - Preserves group-specific distributions while filling missing values

3. Validation Guarantees:
   - Enforces non-negative integer constraints appropriate for attempt counts
   - Clips outliers using group-specific bounds
   - Validates complete removal of NaN values
   - Monitors demographic parity after imputation

4. Fairness Safeguards:
   - Dynamic adjustment of imputation parameters based on group sizes
   - Automatic detection and correction of demographic disparities
   - Continuous monitoring of fairness metrics during imputation

These enhancements ensure:
- Complete elimination of NaN values while preserving fairness
- Maintenance of demographic balance in the imputed data
- Protection against bias amplification during imputation
- Robust handling of edge cases and small demographic groups

### Configuration Parameters

Key fairness parameters that control the preprocessing:

```python
FAIRNESS = {
    'threshold': 0.2,  # Maximum allowed disparity between groups
    'min_group_size': 50,  # Minimum samples required per group
}

BIAS_MITIGATION = {
    'method': 'reweight',  # Options: 'reweight', 'oversample', 'undersample', 'none'
    'balance_strategy': 'group_balanced',
    'max_ratio': 3.0  # Maximum allowed ratio between group sizes
}
```

## Impact on Model Fairness

These preprocessing decisions aim to:
1. Reduce algorithmic bias by standardizing and normalizing features
2. Maintain meaningful demographic patterns while addressing severe imbalances
3. Improve model stability without sacrificing interpretability
4. Support fair ML practices through careful handling of protected attributes

The enhanced preprocessing pipeline helps ensure:

1. Balanced Training Data:
   - More equitable representation of demographic groups
   - Preserved group-specific distributions
   - Reduced impact of systematic biases

2. Fair Feature Engineering:
   - Group-aware feature scaling
   - Protected attribute-sensitive imputation
   - Distribution-preserving transformations

3. Model Input Quality:
   - Reduced demographic skew in features
   - Maintained statistical validity
   - Preserved meaningful group differences

## Validation and Monitoring

The pipeline includes comprehensive fairness validation:

1. Distribution Checks:
   - Before/after comparisons for each transformation
   - Group-specific statistical monitoring
   - Disparity threshold validation

2. Quality Metrics:
   - Missing value impact assessment
   - Group size monitoring
   - Outlier detection with fairness constraints

3. Warning System:
   - Automated alerts for fairness violations
   - Detailed logging of demographic impacts
   - Recommendations for mitigation strategies

## Future Considerations

1. Monitor impact of age band merging on model performance
2. Consider collecting more data for underrepresented groups
3. Evaluate need for more sophisticated balancing techniques if disparities persist