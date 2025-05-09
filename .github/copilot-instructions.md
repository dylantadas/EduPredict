# EduPredict Project - GitHub Copilot Instructions

This document provides guidelines for GitHub Copilot when suggesting code for the EduPredict project, a dual-path ensemble model for academic early warning systems.

## Core Principles

1. **Configuration Consistency**: Always align with parameters defined in `config.py`
2. **Code Reuse**: Avoid duplication by checking existing modules for similar functionality
3. **Architectural Alignment**: Follow the dual-path architecture (Random Forest + GRU ensemble)
4. **Fairness Awareness**: Maintain fairness metrics across demographic groups
5. **Dataset Structure Understanding**: Reference the `./data/README.md` file for data structure context

## Configuration Parameters

Always reference and utilize the parameters defined in `config.py` when suggesting code. This ensures consistency across the codebase and maintains the established configuration standards. Check the current version of this file for the most up-to-date parameters and values.

## Before Adding New Functions

1. **Check for existing implementation**: Search across all modules for similar functions
2. **Look for similar patterns**: Identify functions with similar purposes that could be adapted
3. **Cross-module references**: Be aware of the different modules in the codebase:
   - `data_processing.py`: Data loading and cleaning functions
   - `feature_engineering.py`: Feature creation and transformation
   - `eda.py`: Exploratory data analysis
   - `random_forest_model.py`: Static path implementation
   - `gru_model.py`: Sequential path implementation
   - `ensemble.py`: Model combination logic
   - `performance_metrics.py`: Evaluation metrics
   - `fairness_analysis.py`: Demographic fairness assessment

## Code Structure Guidance

- **Type hints**: Include Python type hints for function arguments and return values
- **Error handling**: Implement appropriate validation and error messages
- **Memory efficiency**: Use techniques like chunking for large datasets

## Project-Specific Considerations

- **Dual-path architecture**: Maintain separation between static and sequential paths
- **Data pipeline**: Preserve data flow: loading → cleaning → feature engineering → modeling
- **Ensemble weighting**: Remember to optimize weights between Random Forest and GRU paths
- **Fairness metrics**: Ensure demographic parity across protected attributes

## Important Paths and Constants

- Data is expected in `./data/OULAD/`
- Output files should go to `./output/`
- Models should be saved to `./output/models/`

## Final Validation

For any significant code suggestion:
1. Verify configuration parameter usage
2. Check for potential code duplication
3. Ensure adherence to the established architecture
4. Consider memory and computational efficiency