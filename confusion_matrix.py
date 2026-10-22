
from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent

GOLD_FILE = folder / "updated_canine_thorax_file.xlsx"

PREDICTIONS_FILE = (
    folder / "gemini_canine_thorax_predictions_100_cases_20_diseases.xlsx"
)

OUTPUT_FILE = folder / "canine_thorax_evaluation_results.xlsx"

DISEASE_COLUMNS = [
    "perihilar_infiltrate",
    "pneumonia",
    "bronchitis",
    "interstitial",
    "diseased_lungs",
    "hypo_plastic_trachea",
    "cardiomegaly",
    "pulmonary_nodules",
    "pleural_effusion",
    "rtm",
    "focal_caudodorsal_lung",
    "focal_perihilar",
    "pulmonary_hypoinflation",
    "right_sided_cardiomegaly",
    "pericardial_effusion",
    "bronchiectasis",
    "pulmonary_vessel_enlargement",
    "left_sided_cardiomegaly",
    "thoracic_lymphadenopathy",
    "esophagitis",
]

def normalize_case_id(value):
    """
    Remove annotation asterisks from CaseIDs.
    Example: '4 *' becomes '4'.
    """

    if pd.isna(value):
        return None

    cleaned = str(value).strip()
    cleaned = cleaned.replace("*", "").strip()

    # Handle IDs such as '4.0'.
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]

    return cleaned


def normalize_labels(series, file_description, column_name):
    """Accept only Positive or Negative labels."""

    cleaned = series.astype("string").str.strip().str.lower()

    mapping = {
        "positive": "Positive",
        "negative": "Negative",
    }

    unexpected = sorted(
        set(cleaned.dropna().unique()) - set(mapping.keys())
    )

    if unexpected:
        raise ValueError(
            f"{file_description}, column '{column_name}' "
            f"contains unexpected labels: {unexpected}"
        )

    normalized = cleaned.map(mapping)

    if normalized.isna().any():
        raise ValueError(
            f"{file_description}, column '{column_name}' "
            "contains blank labels."
        )

    return normalized

def main():

    print("=" * 60)
    print("CANINE THORAX - GEMINI MODEL EVALUATION")
    print("=" * 60)

    # Check input files.
    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Gold Standard file not found:\n{GOLD_FILE}"
        )

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Gemini prediction file not found:\n{PREDICTIONS_FILE}"
        )
    print("\nLoading Excel files...")

    gold = pd.read_excel(GOLD_FILE, sheet_name=0)

    predictions = pd.read_excel(
        PREDICTIONS_FILE,
        sheet_name="Predictions"
    )

    print(f"Gold Standard rows: {len(gold)}")
    print(f"Gemini prediction rows: {len(predictions)}")

    required_columns = ["CaseID"] + DISEASE_COLUMNS

    for description, dataframe in [
        ("Gold Standard", gold),
        ("Gemini Predictions", predictions),
    ]:

        missing_columns = [
            col for col in required_columns
            if col not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{description} is missing columns: "
                f"{missing_columns}"
            )
    gold["CaseID"] = gold["CaseID"].apply(normalize_case_id)

    predictions["CaseID"] = predictions["CaseID"].apply(
        normalize_case_id
    )

    for description, dataframe in [
        ("Gold Standard", gold),
        ("Gemini Predictions", predictions),
    ]:

        if dataframe["CaseID"].isna().any():
            raise ValueError(
                f"{description} contains blank CaseIDs."
            )

        if dataframe["CaseID"].duplicated().any():
            duplicates = dataframe.loc[
                dataframe["CaseID"].duplicated(keep=False),
                "CaseID"
            ].tolist()

            raise ValueError(
                f"{description} contains duplicate CaseIDs: "
                f"{duplicates}"
            )

    gold_ids = set(gold["CaseID"])
    prediction_ids = set(predictions["CaseID"])

    if gold_ids != prediction_ids:

        missing_predictions = sorted(gold_ids - prediction_ids)
        extra_predictions = sorted(prediction_ids - gold_ids)

        raise ValueError(
            "CaseIDs do not match!\n"
            f"Missing from Gemini predictions: "
            f"{missing_predictions}\n"
            f"Extra in Gemini predictions: {extra_predictions}"
        )

    
    gold = gold.sort_values("CaseID").reset_index(drop=True)

    predictions = predictions.sort_values(
        "CaseID"
    ).reset_index(drop=True)

    print("\nCaseIDs match successfully.")
    print("Validating disease labels...")

    for disease in DISEASE_COLUMNS:

        gold[disease] = normalize_labels(
            gold[disease],
            "Gold Standard",
            disease
        )

        predictions[disease] = normalize_labels(
            predictions[disease],
            "Gemini Predictions",
            disease
        )

    print("All disease labels are valid.")
    summary_rows = []
    confusion_rows = []

    for disease in DISEASE_COLUMNS:

        actual = gold[disease]
        predicted = predictions[disease]

        # True Positive
        TP = int(
            ((actual == "Positive") &
             (predicted == "Positive")).sum()
        )

        # False Positive
        FP = int(
            ((actual == "Negative") &
             (predicted == "Positive")).sum()
        )

        # True Negative
        TN = int(
            ((actual == "Negative") &
             (predicted == "Negative")).sum()
        )

        # False Negative
        FN = int(
            ((actual == "Positive") &
             (predicted == "Negative")).sum()
        )

        total = TP + FP + TN + FN

        accuracy = (
            (TP + TN) / total if total else 0.0
        )

        precision = (
            TP / (TP + FP) if (TP + FP) else 0.0
        )

        sensitivity = (
            TP / (TP + FN) if (TP + FN) else 0.0
        )

        specificity = (
            TN / (TN + FP) if (TN + FP) else 0.0
        )

        summary_rows.append({
            "Disease": disease,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "Accuracy": accuracy,
            "Precision": precision,
            "Sensitivity_Recall": sensitivity,
            "Specificity": specificity,
            "Total_Cases": total,
        })

        confusion_rows.extend([
            {
                "Disease": disease,
                "Actual": "Positive",
                "Predicted": "Positive",
                "Count": TP,
            },
            {
                "Disease": disease,
                "Actual": "Positive",
                "Predicted": "Negative",
                "Count": FN,
            },
            {
                "Disease": disease,
                "Actual": "Negative",
                "Predicted": "Positive",
                "Count": FP,
            },
            {
                "Disease": disease,
                "Actual": "Negative",
                "Predicted": "Negative",
                "Count": TN,
            },
        ])

    summary = pd.DataFrame(summary_rows)
    confusion = pd.DataFrame(confusion_rows)
    comparison = pd.DataFrame({
        "CaseID": gold["CaseID"]
    })

    for disease in DISEASE_COLUMNS:

        comparison[f"{disease}_Gold"] = gold[disease]

        comparison[f"{disease}_Gemini"] = predictions[disease]

        comparison[f"{disease}_Match"] = (
            gold[disease] == predictions[disease]
        )
    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        summary.to_excel(
            writer,
            sheet_name="Metrics_Summary",
            index=False
        )

        confusion.to_excel(
            writer,
            sheet_name="Confusion_Counts",
            index=False
        )

        comparison.to_excel(
            writer,
            sheet_name="Case_Comparison",
            index=False
        )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print(f"Gold Standard cases: {len(gold)}")
    print(f"Gemini prediction cases: {len(predictions)}")
    print(f"Diseases evaluated: {len(DISEASE_COLUMNS)}")

    print(f"\nOutput saved to:\n{OUTPUT_FILE}")

    print("\nMETRICS SUMMARY (percentages)")

    display_summary = summary.copy()

    metrics = [
        "Accuracy",
        "Precision",
        "Sensitivity_Recall",
        "Specificity",
    ]

    for metric in metrics:
        display_summary[metric] = (
            display_summary[metric] * 100
        ).round(2)

    print(display_summary.to_string(index=False))

if __name__ == "__main__":
    main()
