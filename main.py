from src.pipeline.svm_part import run_svm_classification
from src.metric.metric import calculate_priority_by_tfidf

if __name__ == "__main__":

    run_svm_classification(calculate_priority_by_tfidf)