"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    brain_data = pd.read_csv(data.csv_path, sep="\t", na_values="NA")
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    men = brain_data[brain_data["Gender"] == "Male"]
    women = brain_data[brain_data["Gender"] == "Female"]

    # Для каждой пары признаков используются строки без пропусков в этой паре.
    men_correlations = men[features + ["MRI_Count"]].corr(method="pearson")["MRI_Count"][features]
    women_correlations = women[features + ["MRI_Count"]].corr(method="pearson")["MRI_Count"][features]
    all_correlations = pd.concat([men_correlations, women_correlations]).dropna()
    if all_correlations.empty:
        raise ValueError("Недостаточно данных для вычисления корреляций с MRI_Count.")

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_correlations.to_dict(),
        men_mri_correlation=men_correlations.to_dict(),
        strongest_mri_feature=str(all_correlations.abs().idxmax()),
    )
