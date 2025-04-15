if __name__ == "__main__":
    # import libraries
    import os
    import sys
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    import warnings
    
    # suppress warnings
    warnings.filterwarnings('ignore')
    
    # set visualization style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    pd.set_option('display.max_columns', None)
    
    # add project root to python path
    project_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    sys.path.insert(0, project_root)
    # import from modules
    from data_processing.data_processing import (
        load_raw_datasets, 
        clean_demographic_data, 
        clean_vle_data, 
        clean_assessment_data, 
        validate_data_consistency
    )
    
    from data_processing.feature_engineering import (
        create_demographic_features,
        create_temporal_features,
        create_assessment_features,
        create_sequential_features,
        prepare_dual_path_features,
        create_stratified_splits,
        prepare_target_variable
    )
    
    from data_analysis.eda import (
        perform_automated_eda,
        analyze_student_performance,
        analyze_engagement_patterns,
        document_eda_findings,
        visualize_demographic_distributions
    )
    
    from evaluation.performance_metrics import (
        analyze_feature_importance,
        analyze_feature_correlations,
        calculate_model_metrics,
        calculate_fairness_metrics,
        plot_roc_curves,
        plot_fairness_metrics
    )
    
    from model_training.hyperparameter_tuning import (
        tune_random_forest,
        visualize_tuning_results
    )
    # current working directory
    print("Working directory:", os.getcwd())
    
    # check project root data directory
    project_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    data_path = os.path.join(project_root, "data", "OULAD")
    print("Project root data path:", data_path)
    
    # list contents
    if os.path.exists(data_path):
        print("Contents:", os.listdir(data_path))
    # load datasets using optimized loading function
    try:
        # load datasets
        datasets = load_raw_datasets(data_path)
        print("Dataset keys:", list(datasets.keys()))
    
        # verify data consistency
        if validate_data_consistency(datasets):
            print("Data consistency validation passed.")
        else:
            print("Data consistency validation failed.")
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
    # display dataset shapes
    for name, df in datasets.items():
        print(f"{name}: {df.shape}")
    # clean demographic data
    clean_demographics = clean_demographic_data(datasets['student_info'])
    print(f"Clean demographics shape: {clean_demographics.shape}")
    
    # clean vle data
    clean_vle = clean_vle_data(datasets['vle_interactions'], datasets['vle_materials'])
    print(f"Clean VLE data shape: {clean_vle.shape}")
    
    # clean assessment data
    clean_assessments = clean_assessment_data(datasets['assessments'], datasets['student_assessments'])
    print(f"Clean assessment data shape: {clean_assessments.shape}")
    # run automated eda on clean datasets
    clean_datasets = {
        'student_info': clean_demographics,
        'vle': clean_vle,
        'assessments': clean_assessments
    }
    
    # get documented findings
    eda_findings = document_eda_findings(clean_datasets)
    # visualize demographic distributions
    visualize_demographic_distributions(clean_demographics)
    # examine performance patterns by demographic group
    # combine demographic and assessment data
    demo_assessment = clean_demographics.merge(
        clean_assessments,
        on=['id_student', 'code_module', 'code_presentation'],
        how='inner'
    )
    
    # analyze performance across demographic groups
    for col in ['gender', 'age_band', 'imd_band']:
        if col in demo_assessment.columns:
            print(f"\nAverage score by {col}:")
            print(demo_assessment.groupby(col, observed=False)['score'].mean().sort_values())
    # examine final result distribution
    if 'final_result' in clean_demographics.columns:
        plt.figure(figsize=(10, 6))
        result_counts = clean_demographics['final_result'].value_counts()
        result_counts.plot(kind='bar')
        plt.title('Distribution of Final Results')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # add percentage labels
        total = result_counts.sum()
        for i, count in enumerate(result_counts):
            plt.text(i, count + 100, f'{100 * count / total:.1f}%', ha='center')
        
        plt.show()
    # create demographic features
    demographic_features = create_demographic_features(clean_demographics)
    print(f"Demographic features shape: {demographic_features.shape}")
    
    # display sample of demographic features
    demographic_features.head()
    # create temporal features with multiple window sizes
    window_sizes = [7, 14, 30]  # weekly, bi-weekly, monthly
    temporal_features = create_temporal_features(clean_vle, window_sizes)
    
    # display info about temporal features
    for window_size, features in temporal_features.items():
        print(f"{window_size} features shape: {features.shape}")
    # create assessment features
    assessment_features = create_assessment_features(clean_assessments)
    print(f"Assessment features shape: {assessment_features.shape}")
    
    # display sample of assessment features
    assessment_features.head()
    # create sequential features for gru path
    sequential_features = create_sequential_features(clean_vle)
    print(f"Sequential features shape: {sequential_features.shape}")
    # prepare dual path features
    dual_path_features = prepare_dual_path_features(
        demographic_features, 
        temporal_features,
        assessment_features,
        sequential_features
    )
    
    # check shapes of dual path features
    for path_name, features in dual_path_features.items():
        print(f"{path_name} shape: {features.shape}")
    # create binary target variable
    static_features = dual_path_features['static_path']
    y = prepare_target_variable(static_features)
    
    # check target distribution
    target_counts = y.value_counts()
    print("Target distribution:")
    print(target_counts)
    print(f"Percentage at risk: {100 * target_counts.get(1, 0) / len(y):.2f}%")
    # create stratified splits
    split_data = create_stratified_splits(dual_path_features, test_size=0.2, random_state=42)
    # prepare data for static path modeling
    X_train_static = split_data['static_train'].drop(['final_result', 'id_student', 'code_module', 'code_presentation'], axis=1, errors='ignore')
    X_test_static = split_data['static_test'].drop(['final_result', 'id_student', 'code_module', 'code_presentation'], axis=1, errors='ignore')
    
    # prepare target variables
    y_train = prepare_target_variable(split_data['static_train'])
    y_test = prepare_target_variable(split_data['static_test'])
    
    # display shapes
    print(f"X_train shape: {X_train_static.shape}")
    print(f"X_test shape: {X_test_static.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    # analyze feature importance
    feature_importance = analyze_feature_importance(X_train_static, y_train)
    # identify highly correlated features
    correlated_features = analyze_feature_correlations(X_train_static, threshold=0.85)
    
    if len(correlated_features) > 0:
        print("\nHighly correlated features:")
        print(correlated_features)
    else:
        print("\nNo highly correlated features found (threshold: 0.85).")
    # feature selection based on importance
    importance_threshold = 0.01
    important_features = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature'].tolist()
    print(f"Selected {len(important_features)} features with importance > {importance_threshold}")
    
    # filter to important features
    X_train_selected = X_train_static[important_features]
    X_test_selected = X_test_static[important_features]
    
    print(f"X_train_selected shape: {X_train_selected.shape}")
    print(f"X_test_selected shape: {X_test_selected.shape}")
    # train baseline random forest model
    baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    baseline_rf.fit(X_train_selected, y_train)
    
    # make predictions
    y_pred = baseline_rf.predict(X_test_selected)
    y_prob = baseline_rf.predict_proba(X_test_selected)[:, 1]
    
    # display classification report
    print("Classification Report (Baseline Random Forest):")
    print(classification_report(y_test, y_pred))
    
    # display confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # ROC AUC score
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {auc:.4f}")
    # define parameter grid for random forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # tune random forest hyperparameters
    best_params, best_model = tune_random_forest(
        X_train_selected, 
        y_train,
        param_grid=param_grid,
        scoring='f1',
        random_search=True,
        n_iter=20,
        verbose=1
    )
    # evaluate optimized model
    y_pred_opt = best_model.predict(X_test_selected)
    y_prob_opt = best_model.predict_proba(X_test_selected)[:, 1]
    
    # display classification report
    print("Classification Report (Optimized Random Forest):")
    print(classification_report(y_test, y_pred_opt))
    
    # display confusion matrix
    print("\nConfusion Matrix:")
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    print(cm_opt)
    
    # ROC AUC score
    auc_opt = roc_auc_score(y_test, y_prob_opt)
    print(f"\nROC AUC Score: {auc_opt:.4f}")
    # plot ROC curve
    plot_roc_curves(y_test, y_prob_opt)
    # prepare protected attributes for fairness analysis
    protected_attributes = {}
    
    for attr in ['gender', 'age_band', 'imd_band']:
        if attr in split_data['static_test'].columns:
            protected_attributes[attr] = split_data['static_test'][attr].values
    
    # calculate fairness metrics
    fairness_results = calculate_fairness_metrics(
        y_test.values, 
        y_pred_opt, 
        y_prob_opt,
        protected_attributes
    )
    
    # display fairness metrics
    for attr, metrics in fairness_results.items():
        print(f"\nFairness metrics for {attr}:")
        if 'demographic_parity_difference' in metrics:
            print(f"Demographic parity difference: {metrics['demographic_parity_difference']:.4f}")
        if 'disparate_impact_ratio' in metrics:
            print(f"Disparate impact ratio: {metrics['disparate_impact_ratio']:.4f}")
        if 'equal_opportunity_difference' in metrics:
            print(f"Equal opportunity difference: {metrics['equal_opportunity_difference']:.4f}")
    # plot fairness metrics by demographic group
    plot_fairness_metrics(fairness_results, metric_name='f1')
    # plot ROC curves by gender
    if 'gender' in protected_attributes:
        plot_roc_curves(y_test.values, y_prob_opt, protected_attributes['gender'], group_name='Gender')
    # plot ROC curves by IMD band
    if 'imd_band' in protected_attributes:
        plot_roc_curves(y_test.values, y_prob_opt, protected_attributes['imd_band'], group_name='IMD Band')
    import joblib
    import os
    
    # create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # save optimized random forest model
    joblib.dump(best_model, '../models/random_forest_optimized.pkl')
    print("Saved optimized Random Forest model.")
    
    # save important features list
    pd.Series(important_features).to_csv('../models/important_features.csv', index=False)
    print("Saved important features list.")
    
    # save feature importance data
    feature_importance.to_csv('../models/feature_importance.csv', index=False)
    print("Saved feature importance data.")