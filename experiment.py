# =============================================================================
# Rule-Based AI: Credit Risk Assessment
# Comparing Strict Cut-Offs vs. Calibrated Weighted Rules with Compensating Factors
#
# AI Assistance Note: Generative AI (Google Gemini) was used to assist with
# code structuring, threshold-calibration implementation, debugging, and
# Matplotlib chart formatting. All experimental design decisions, rules,
# weight assignments, threshold selections, result interpretations, and final
# outputs were reviewed and verified by the author.
# =============================================================================

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# 1. Load, Clean, and Split Dataset (Leakage-Free Preprocessing)
def load_data(filepath="credit_risk_dataset.csv"):
    # Note: If running on Kaggle, change filepath to:
    # "/kaggle/input/credit-risk-dataset/credit_risk_dataset.csv"
    df = pd.read_csv(filepath)
    raw_count = len(df)

    # Remove extreme data-entry outliers while preserving NaNs for imputation
    df = df[df["person_age"].isna() | (df["person_age"] <= 100)].copy()
    df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)].copy()
    cleaned_count = len(df)

    # 80/20 Stratified Split BEFORE imputation to prevent data leakage
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["loan_status"]
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Compute medians strictly on the 80% calibration split and apply to both splits
    emp_median = train_df["person_emp_length"].median()
    rate_median = train_df["loan_int_rate"].median()

    train_df["person_emp_length"] = train_df["person_emp_length"].fillna(emp_median)
    train_df["loan_int_rate"] = train_df["loan_int_rate"].fillna(rate_median)
    test_df["person_emp_length"] = test_df["person_emp_length"].fillna(emp_median)
    test_df["loan_int_rate"] = test_df["loan_int_rate"].fillna(rate_median)

    print("--- Dataset Summary ---")
    print(f"Raw Records        : {raw_count}")
    print(
        f"Cleaned Records    : {cleaned_count} (Outliers removed: {raw_count - cleaned_count})"
    )
    print(f"Calibration Split  : {len(train_df)} (80%)")
    print(f"Evaluation Holdout : {len(test_df)} (20%)")
    print(f"Test Default Rate  : {test_df['loan_status'].mean() * 100:.2f}%\n")

    return train_df, test_df


# 2. Algorithm 1: Deterministic Forward-Chaining Heuristic System
def algorithm_1_strict_rules(row):
    if row["cb_person_default_on_file"] == "Y":
        return 1
    if row["loan_percent_income"] > 0.30:
        return 1
    if row["loan_grade"] in ["D", "E", "F", "G"]:
        return 1
    return 0


# 3. Algorithm 2: Weighted Multi-Factor Scoring with Compensating Factors
def calculate_risk_score(row):
    risk_score = 0

    # --- Risk Factors (Positive Points) ---
    # Debt-to-income repayment strain
    if row["loan_percent_income"] > 0.35:
        risk_score += 45
    elif row["loan_percent_income"] > 0.30:
        risk_score += 35
    elif row["loan_percent_income"] >= 0.25:
        risk_score += 15

    # Credit grade risk tier
    grade = row["loan_grade"]
    if grade == "G":
        risk_score += 55
    elif grade == "F":
        risk_score += 45
    elif grade == "E":
        risk_score += 40
    elif grade == "D":
        risk_score += 32
    elif grade == "C":
        risk_score += 10

    # Interest rate risk premium
    if row["loan_int_rate"] >= 15.0:
        risk_score += 15
    elif row["loan_int_rate"] > 13.5:
        risk_score += 8

    # Prior historical default flag
    if row["cb_person_default_on_file"] == "Y":
        risk_score += 10

    # Housing vulnerability (no home equity buffer)
    if row["person_home_ownership"] in ["RENT", "OTHER"]:
        risk_score += 15

    # Low income cushion
    if row["person_income"] < 25000:
        risk_score += 20
    elif row["person_income"] < 40000:
        risk_score += 10

    # Higher-risk loan purposes
    if row["loan_intent"] in ["DEBTCONSOLIDATION", "MEDICAL", "HOMEIMPROVEMENT"]:
        risk_score += 8

    # Newly employed / zero tenure
    if row["person_emp_length"] == 0:
        risk_score += 5

    # --- Compensating Stability Factors (Negative Points) ---
    # Home equity / mortgage vetting
    if row["person_home_ownership"] == "OWN":
        risk_score -= 22
    elif row["person_home_ownership"] == "MORTGAGE":
        risk_score -= 12

    # Sustained employment history
    if row["person_emp_length"] >= 5:
        risk_score -= 8

    # High annual income buffer
    if row["person_income"] >= 80000:
        risk_score -= 10

    # Lower-default loan purposes
    if row["loan_intent"] in ["VENTURE", "EDUCATION"]:
        risk_score -= 8

    return risk_score


def algorithm_2_weighted_rules(row, threshold=35):
    return 1 if calculate_risk_score(row) >= threshold else 0


# 4. Heuristic Threshold Calibration (Strictly on 80% Calibration Split)
def calibrate_threshold(calibration_df, candidate_thresholds=range(20, 61, 5)):
    scores = calibration_df.apply(calculate_risk_score, axis=1)
    y_true = calibration_df["loan_status"]

    best_threshold = 35
    best_f1 = -1.0

    print("--- Algorithm 2 Threshold Calibration (80% Calibration Split) ---")
    print(
        f"{'Threshold':>9} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'FPR':>6} | {'FNR':>6}"
    )
    print("-" * 62)

    for thresh in candidate_thresholds:
        y_pred = (scores >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        print(
            f"{thresh:>9} | {acc:>6.4f} | {prec:>6.4f} | {rec:>6.4f} | "
            f"{f1:>6.4f} | {fpr:>6.4f} | {fnr:>6.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(
        f"\nSelected Optimal Threshold (Max Calibration F1): >= {best_threshold} (F1 = {best_f1:.4f})\n"
    )
    return best_threshold


# 5. Evaluation and Speed Benchmarking (Strictly on 20% Unseen Holdout)
def evaluate_system(predict_fn, data, name="Algorithm"):
    start = time.perf_counter()
    y_pred = data.apply(predict_fn, axis=1)
    end = time.perf_counter()

    y_true = data["loan_status"]
    latency_ms_per_rec = ((end - start) / len(data)) * 1000

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Type 1 Error Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Type 2 Error Rate

    print(f"--- {name} ---")
    print(f"Confusion Matrix   : TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print(f"Accuracy           : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision          : {prec:.4f}")
    print(f"Recall             : {rec:.4f}")
    print(f"F1-Score           : {f1:.4f}")
    print(f"Type 1 Error (FPR) : {fpr:.4f} ({fpr * 100:.2f}%)")
    print(f"Type 2 Error (FNR) : {fnr:.4f} ({fnr * 100:.2f}%)")
    print(f"Latency            : {latency_ms_per_rec:.4f} ms/record\n")

    return {
        "name": name,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "cm": (tn, fp, fn, tp),
        "latency": latency_ms_per_rec,
    }


# 6. Single Applicant Input Demonstration (Auditability Check)
def test_single_applicant(applicant_data, optimal_threshold):
    row = pd.Series(applicant_data)
    score = calculate_risk_score(row)

    res1 = (
        "REJECT (High Risk)"
        if algorithm_1_strict_rules(row) == 1
        else "APPROVE (Low Risk)"
    )
    res2 = (
        "REJECT (High Risk)"
        if algorithm_2_weighted_rules(row, threshold=optimal_threshold) == 1
        else "APPROVE (Low Risk)"
    )

    print("--- Single Applicant Input Test ---")
    print(
        f"Input Profile : Income=${applicant_data['person_income']}, "
        f"Loan=${applicant_data['loan_amnt']}, "
        f"DTI={applicant_data['loan_percent_income']:.2f}, "
        f"Grade={applicant_data['loan_grade']}, "
        f"Home={applicant_data['person_home_ownership']}, "
        f"Emp={applicant_data['person_emp_length']} yrs, "
        f"Intent={applicant_data['loan_intent']}"
    )
    print(f"Algorithm 1 Decision: {res1}")
    print(
        f"Algorithm 2 Decision: {res2} "
        f"(Audit Score: {score} vs. Threshold: >= {optimal_threshold})\n"
    )


# 7. Main Execution and Visualization
if __name__ == "__main__":
    train_data, test_data = load_data()

    # Calibrate Algorithm 2 threshold strictly on the 80% calibration split
    optimal_threshold = calibrate_threshold(train_data)

    # Evaluate both algorithms on the unseen 20% holdout set
    res1 = evaluate_system(
        algorithm_1_strict_rules,
        test_data,
        "Algorithm 1 (Strict Forward-Chaining)",
    )
    res2 = evaluate_system(
        lambda row: algorithm_2_weighted_rules(row, threshold=optimal_threshold),
        test_data,
        f"Algorithm 2 (Calibrated Weighted Rules, Cutoff >= {optimal_threshold})",
    )

    # Summarise Type 1 Error (False Positive) Reduction
    fp1, fp2 = res1["cm"][1], res2["cm"][1]
    fp_saved = fp1 - fp2
    fp_reduction_pct = (fp_saved / fp1 * 100) if fp1 > 0 else 0.0
    print("--- Type 1 Error (False Rejection) Impact ---")
    print(
        f"False Positives Reduced: {fp1} -> {fp2} "
        f"({fp_saved} safe borrowers saved, -{fp_reduction_pct:.2f}%)\n"
    )

    # Single applicant test showing how compensating factors prevent an unfair rejection
    sample_input = {
        "person_income": 45000,
        "loan_amnt": 13500,
        "loan_percent_income": 0.30,
        "loan_int_rate": 13.5,
        "cb_person_default_on_file": "N",
        "loan_grade": "D",
        "person_home_ownership": "OWN",
        "person_emp_length": 7.0,
        "loan_intent": "PERSONAL",
    }
    test_single_applicant(sample_input, optimal_threshold)

    # Dual-panel Matplotlib layout formatted with assistance from Gemini (Google, 2026)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), dpi=300, gridspec_kw={"width_ratios": [2.2, 1.4]}
    )

    # Panel A: Standard Classification Metrics (Higher is Better)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    vals1 = [res1["acc"], res1["prec"], res1["rec"], res1["f1"]]
    vals2 = [res2["acc"], res2["prec"], res2["rec"], res2["f1"]]

    x1 = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(
        x1 - width / 2,
        vals1,
        width,
        label="Algorithm 1: Strict Forward-Chaining",
        color="#1f77b4",
    )
    bars2 = ax1.bar(
        x1 + width / 2,
        vals2,
        width,
        label=f"Algorithm 2: Weighted Rules (>= {optimal_threshold})",
        color="#2ca02c",
    )

    ax1.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax1.set_title(
        "Classification Performance (Higher is Better)", fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(x1)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.set_ylim(0, 1.12)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel B: Error Rates — Type 1 (FPR) and Type 2 (FNR) (Lower is Better)
    error_labels = ["Type 1 Error\n(FPR)", "Type 2 Error\n(FNR)"]
    err1 = [res1["fpr"] * 100, res1["fnr"] * 100]
    err2 = [res2["fpr"] * 100, res2["fnr"] * 100]

    x2 = np.arange(len(error_labels))
    ebars1 = ax2.bar(x2 - width / 2, err1, width, label="Algorithm 1", color="#1f77b4")
    ebars2 = ax2.bar(x2 + width / 2, err2, width, label="Algorithm 2", color="#2ca02c")

    ax2.set_ylabel("Error Rate (%)", fontsize=11)
    ax2.set_title(
        "Error Rate Comparison (Lower is Better)", fontsize=12, fontweight="bold"
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(error_labels, fontsize=10)
    max_err = max(err1 + err2)
    ax2.set_ylim(0, max_err * 1.25 if max_err > 0 else 10)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    for bar in list(ebars1) + list(ebars2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300)
    print("Saved high-resolution chart: model_comparison.png")
