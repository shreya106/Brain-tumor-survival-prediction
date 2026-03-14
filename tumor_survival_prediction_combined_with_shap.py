# Full pipeline for Glioma tumor classification and survival prediction
# Includes preprocessing, SMOTE, feature selection, ensemble models, SVM comparison, and SHAP analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ------------------------
# Data Loading & Cleaning
# ------------------------
print("Loading and preprocessing dataset...")

xls = pd.ExcelFile("Glioma-clinic-TCGA.xlsx")
df = xls.parse('final data', skiprows=1)
df = df[df['histological_type'].notna() & df['outcome'].notna()]
df['outcome_encoded'] = df['outcome'].astype(int)

# Drop columns that are not numeric or relevant for modeling
drop_cols = ['Case', 'gender', 'histological_type', 'race', 'ethnicity',
             'radiation_therapy', 'Grade', 'outcome', 'IDH.status']
X = df.drop(columns=drop_cols).apply(pd.to_numeric, errors='coerce')
X = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X), columns=X.columns)

# Encode tumor classification target
y_tumor = LabelEncoder().fit_transform(df['histological_type'])
# Survival prediction target
y_surv = df['outcome_encoded']

# Display class distributions
plt.figure(figsize=(8, 4))
sns.countplot(x=df['histological_type'])
plt.title('Tumor Type Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('tumor_type_distribution.png')
plt.close()

plt.figure(figsize=(5, 4))
sns.countplot(x=y_surv)
plt.title('Survival Class Distribution (0=Deceased, 1=Living)')
plt.tight_layout()
plt.savefig('survival_class_distribution.png')
plt.close()

# ------------------------
# Tumor Classification
# ------------------------

X_resampled, y_resampled = SMOTEENN(random_state=42).fit_resample(X, y_tumor)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

selector = SelectKBest(score_func=f_classif, k=150)
X_train_sel = selector.fit_transform(X_train_poly, y_train)
X_test_sel = selector.transform(X_test_poly)
selected_columns_tumor = selector.get_feature_names_out()

# Stacking model setup
base_models = [
    ('rf', RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, class_weight='balanced')),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
]
stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1)

print("\n Training StackingClassifier for tumor classification...")
stack_model.fit(X_train_sel, y_train)
# Encode tumor classification labels
le_tumor = LabelEncoder()
y_tumor = le_tumor.fit_transform(df['histological_type'])
y_pred = stack_model.predict(X_test_sel)

acc_tumor = accuracy_score(y_test, y_pred)
print("\n Tumor Classification Accuracy: {:.2f}%".format(acc_tumor * 100))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix - Tumor
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(pd.DataFrame(cm), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Tumor Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_stacking_classifier.png")
plt.close()

# ------------------------
# Survival Prediction
# ------------------------

X_train_surv, X_test_surv, y_train_surv, y_test_surv = train_test_split(X, y_surv, test_size=0.2, stratify=y_surv, random_state=42)
X_train_surv_resampled, y_train_surv_resampled = SMOTE(random_state=42).fit_resample(X_train_surv, y_train_surv)
X_train_surv_resampled = pd.DataFrame(X_train_surv_resampled, columns=X_train_surv.columns)

# Feature Selection
rf_surv_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_surv_selector.fit(X_train_surv_resampled, y_train_surv_resampled)
surv_selected = rf_surv_selector.feature_importances_ > np.percentile(rf_surv_selector.feature_importances_, 70)
X_train_surv_selected = X_train_surv_resampled.loc[:, surv_selected]
X_test_surv_selected = X_test_surv.loc[:, surv_selected]
selected_columns_surv = X_train_surv_selected.columns.tolist()

# Train survival models
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_surv_selected, y_train_surv_resampled)
y_pred_surv_lr = lr_model.predict(X_test_surv_selected)
surv_lr_accuracy = accuracy_score(y_test_surv, y_pred_surv_lr)

rf_surv_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rf_surv_model.fit(X_train_surv_selected, y_train_surv_resampled)
y_pred_surv_rf = rf_surv_model.predict(X_test_surv_selected)
surv_rf_accuracy = accuracy_score(y_test_surv, y_pred_surv_rf)

svm_model = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')
svm_model.fit(X_train_surv_selected, y_train_surv_resampled)
y_pred_surv_svm = svm_model.predict(X_test_surv_selected)
surv_svm_accuracy = accuracy_score(y_test_surv, y_pred_surv_svm)

surv_voting_clf = VotingClassifier(estimators=[('rf', rf_surv_model), ('lr', lr_model)], voting='soft')
surv_voting_clf.fit(X_train_surv_selected, y_train_surv_resampled)
y_pred_surv_ensemble = surv_voting_clf.predict(X_test_surv_selected)
surv_ensemble_accuracy = accuracy_score(y_test_surv, y_pred_surv_ensemble)

# Compare and select best model
all_models = {
    'LogisticRegression': (lr_model, surv_lr_accuracy),
    'RandomForest': (rf_surv_model, surv_rf_accuracy),
    'VotingEnsemble': (surv_voting_clf, surv_ensemble_accuracy),
    'SVM': (svm_model, surv_svm_accuracy)
}
final_surv_model_name, (final_surv_model, best_surv_accuracy) = max(all_models.items(), key=lambda x: x[1][1])
best_surv_preds = {
    'LogisticRegression': y_pred_surv_lr,
    'RandomForest': y_pred_surv_rf,
    'VotingEnsemble': y_pred_surv_ensemble,
    'SVM': y_pred_surv_svm
}[final_surv_model_name]

# Confusion Matrix - Survival
cm_surv = confusion_matrix(y_test_surv, best_surv_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(pd.DataFrame(cm_surv), annot=True, fmt="d", cmap="Greens", xticklabels=['Deceased', 'Living'], yticklabels=['Deceased', 'Living'])
plt.title(f"Confusion Matrix - Best Survival Model ({final_surv_model_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_survival.png")
plt.close()

# ------------------------
# SHAP Analysis
# ------------------------

# SHAP - Tumor
print("\n SHAP Analysis for Tumor Classification")
explainer_tumor = shap.Explainer(stack_model.named_estimators_['rf'], X_test_sel)
shap_values_tumor = explainer_tumor(X_test_sel)
shap.summary_plot(shap_values_tumor, X_test_sel, feature_names=selected_columns_tumor, max_display=15, show=False)
plt.title("SHAP Summary - Tumor Classification")
plt.tight_layout()
plt.savefig("shap_summary_tumor.png")
plt.close()

# SHAP - Survival (Improved)
print("\n Improved SHAP Analysis for Survival Prediction")

try:
    if isinstance(final_surv_model, VotingClassifier):
        shap_model = final_surv_model.named_estimators_['rf']
    elif isinstance(final_surv_model, RandomForestClassifier):
        shap_model = final_surv_model
    else:
        shap_model = None  # fallback

    if shap_model:
        explainer_surv = shap.TreeExplainer(shap_model)
        shap_values_surv = explainer_surv.shap_values(X_test_surv_selected)
        shap.summary_plot(shap_values_surv, X_test_surv_selected, feature_names=selected_columns_surv, max_display=15, show=False)
        plt.title("SHAP Summary - Survival Prediction")
        plt.tight_layout()
        plt.savefig("shap_summary_survival.png")
        plt.close()
        print("Improved SHAP summary plot saved: shap_summary_survival.png")
    else:
        print(" SHAP TreeExplainer not compatible with final_surv_model, skipping plot.")
except Exception as e:
    print("SHAP summary plot failed:", str(e))


# ------------------------
# Final Results Summary
# ------------------------

print("\n FINAL RESULTS")
print(f"\n Tumor Classification Accuracy: {acc_tumor:.2%}")
print(" Survival Model Accuracies:")
for name, (_, acc) in all_models.items():
    print(f"  - {name}: {acc:.2%}")
print(f"\n Best Survival Model: {final_surv_model_name} ({best_surv_accuracy:.2%})")
print("\nPlots saved:")
print(" - tumor_type_distribution.png")
print(" - survival_class_distribution.png")
print(" - confusion_matrix_stacking_classifier.png")
print(" - confusion_matrix_survival.png")
print(" - shap_summary_tumor.png")
print(" - shap_summary_survival.png")

# ------------------------
# SHAP Bar Plot Analysis Only
# ------------------------

print("\nSHAP Bar Plot: Tumor Classification")
try:
    X_test_sel_df = pd.DataFrame(X_test_sel, columns=selected_columns_tumor)
    tumor_rf_model = stack_model.named_estimators_['rf']
    explainer_tumor = shap.TreeExplainer(tumor_rf_model)
    shap_values_tumor = explainer_tumor.shap_values(X_test_sel_df)

    if isinstance(shap_values_tumor, list):
        shap_vals_tumor = shap_values_tumor[1] if len(shap_values_tumor) > 1 else shap_values_tumor[0]
    else:
        shap_vals_tumor = shap_values_tumor

    # Bar plot only
    shap.summary_plot(shap_vals_tumor, X_test_sel_df, plot_type="bar", max_display=15, show=False)
    plt.title("Top Features - Tumor Classification")
    plt.tight_layout()
    plt.savefig("shap_barplot_tumor_combined.png")
    plt.close()
    print("SHAP bar plot saved for tumor classification.")
except Exception as e:
    print("SHAP analysis for tumor failed:", str(e))

# ------------------------
# Feature Importance for Survival Prediction
# ------------------------

print("\nFeature Importance: Survival Prediction")
try:
    if hasattr(final_surv_model, "feature_importances_"):
        # Either direct RF model or the RF component of the ensemble
        if isinstance(final_surv_model, VotingClassifier):
            surv_feature_model = final_surv_model.named_estimators_['rf']
        else:
            surv_feature_model = final_surv_model

        surv_importance_df = pd.DataFrame({
            'Feature': selected_columns_surv,
            'Importance': surv_feature_model.feature_importances_
        })
        surv_importance_df = surv_importance_df.sort_values('Importance', ascending=False)
    else:
        # Fallback to permutation importance
        result = permutation_importance(
            final_surv_model, X_test_surv_selected, y_test_surv,
            n_repeats=5,
            random_state=42
        )
        surv_importance_df = pd.DataFrame({
            'Feature': selected_columns_surv,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=surv_importance_df.head(15))
    plt.title('Top 15 Features for Survival Prediction')
    plt.tight_layout()
    plt.savefig('survival_prediction_feature_importance.png')
    plt.close()
    print("Feature importance plot saved: survival_prediction_feature_importance.png")
except Exception as e:
    print("Feature importance analysis for survival failed:", str(e))


    # ------------------------
# Feature Importance for Tumor Classification
# ------------------------

print("\nFeature Importance: Tumor Classification")
try:
    if hasattr(stack_model.named_estimators_['rf'], "feature_importances_"):
        tumor_feature_model = stack_model.named_estimators_['rf']

        tumor_importance_df = pd.DataFrame({
            'Feature': selected_columns_tumor,
            'Importance': tumor_feature_model.feature_importances_
        })
        tumor_importance_df = tumor_importance_df.sort_values('Importance', ascending=False)
    else:
        # Fallback to permutation importance
        result = permutation_importance(
            stack_model, X_test_sel, y_test,
            n_repeats=5,
            random_state=42
        )
        tumor_importance_df = pd.DataFrame({
            'Feature': selected_columns_tumor,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=tumor_importance_df.head(15))
    plt.title('Top 15 Features for Tumor Classification')
    plt.tight_layout()
    plt.savefig('tumor_classification_feature_importance.png')
    plt.close()
    print("Feature importance plot saved: tumor_classification_feature_importance.png")
except Exception as e:
    print("Feature importance analysis for tumor failed:", str(e))