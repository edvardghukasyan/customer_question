import pandas as pd
import logging
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from configs.config import (
    CSV_OUTPUT_PATH,
    CSV_FILE_PATH,
    EVALUATION_RESULTS_CSV,
    EVALUATION_RESULTS_JSON
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_and_update(
    pred_csv: str = CSV_OUTPUT_PATH,
    real_csv: str = CSV_FILE_PATH,
    output_csv: str = EVALUATION_RESULTS_CSV,
    output_json: str = EVALUATION_RESULTS_JSON
):

    try:
        df_pred = pd.read_csv(pred_csv)
        logger.info("Loaded predicted results from '%s'", pred_csv)
    except Exception as e:
        logger.error("Error loading predicted CSV '%s': %s", pred_csv, e)
        raise
    
    try:
        df_real = pd.read_csv(real_csv)
        logger.info("Loaded real data from '%s'", real_csv)
    except Exception as e:
        logger.error("Error loading real CSV '%s': %s", real_csv, e)
        raise
    
    try:
        df_merged = pd.merge(
            df_real[["instruction", "intent"]],
            df_pred,
            how="inner",
            left_on="instruction",
            right_on="question"
        )
        df_merged.rename(columns={"instruction": "question_instruction"}, inplace=True)
        df_merged = df_merged[["question_instruction", "intent", "cluster_name", "description"]]
        logger.info("Merged data shape: %s", df_merged.shape)
    except Exception as e:
        logger.error("Error merging data: %s", e)
        raise
    
    try:
        majority_mapping = (
            df_merged.groupby("cluster_name")["intent"]
            .agg(lambda x: x.value_counts().idxmax())
            .to_dict()
        )
        logger.info("Computed majority mapping: %s", majority_mapping)
    except Exception as e:
        logger.error("Error computing majority mapping: %s", e)
        raise
    
    try:
        confusion = pd.crosstab(df_merged["cluster_name"], df_merged["intent"])
        logger.info("Confusion matrix:\n%s", confusion)
    except Exception as e:
        logger.error("Error computing confusion matrix: %s", e)
        raise
    
    try:
        cost_matrix = -confusion.values
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        predicted_clusters = confusion.index.tolist()
        true_labels = confusion.columns.tolist()
        assignment_map = {predicted_clusters[i]: true_labels[j] for i, j in zip(row_ind, col_ind)}
        logger.info("Hungarian assignment mapping: %s", assignment_map)
    except Exception as e:
        logger.error("Error during assignment mapping: %s", e)
        raise
    

    try:
        df_merged["mapped_intent"] = df_merged["cluster_name"].map(assignment_map)
        df_merged["mapped_intent"].fillna(df_merged["cluster_name"].map(majority_mapping), inplace=True)
    except Exception as e:
        logger.error("Error mapping predicted clusters: %s", e)
        raise
    
    try:
        accuracy = accuracy_score(df_merged["intent"], df_merged["mapped_intent"])
        logger.info("Overall Accuracy (after mapping all points): %.4f", accuracy)
    except Exception as e:
        logger.error("Error computing accuracy: %s", e)
    
    try:
        df_count = df_merged.groupby("cluster_name").size().reset_index(name="count")
        logger.info("Computed counts per cluster:\n%s", df_count.to_string(index=False))
    except Exception as e:
        logger.error("Error computing counts: %s", e)
        raise
    
    try:
        df_pred_unique = df_pred.drop_duplicates(subset=["cluster_name"])
        final_output = pd.merge(df_pred_unique, df_count, on="cluster_name", how="left")
        final_output = final_output[["cluster_name", "description", "count"]].rename(columns={"cluster_name": "name"})
        logger.info("Final aggregated data:\n%s", final_output.to_string(index=False))
    except Exception as e:
        logger.error("Error merging count into predicted data: %s", e)
        raise
    
    try:
        final_output.to_csv(output_csv, index=False)
        logger.info("Final aggregated results saved to CSV: %s", output_csv)
    except Exception as e:
        logger.error("Error saving aggregated CSV '%s': %s", output_csv, e)
    
    try:
        final_output.to_json(output_json, orient="records", indent=4)
        logger.info("Final aggregated results saved to JSON: %s", output_json)
    except Exception as e:
        logger.error("Error saving aggregated JSON '%s': %s", output_json, e)
    
    return final_output


if __name__ == "__main__":
    final_df = evaluate_and_update()
    logger.info("Sample final aggregated data:\n%s", final_df.head().to_string(index=False))