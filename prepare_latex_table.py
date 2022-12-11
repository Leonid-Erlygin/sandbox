import pandas as pd
import numpy as np

DATASET_NAME = "ПК"

if DATASET_NAME == "ЖК":

    METRIC_DATA_PATH = "/home/leonid/work/Predictive-Validation/outputs/Dataset_mortgages_short_metrics.csv"
elif DATASET_NAME == "ПК":
    METRIC_DATA_PATH = "/home/leonid/work/Predictive-Validation/outputs/Dataset_loans_short_metrics.csv"

column_name_to_latex_name = {
    "ПК": {
        "model": "Модель",
        "future horizon months": (
            "Прогн.",
            "(мес.)",
        ),  # first name part will be on first line of table header, second part will be on second line
        "MAE": "MAE",
        "RMSE": ("RMSE", "$\cdot 10^3$"),
        "R2": "R2",
        "Root-mean-squared Calibration Error": "RMSCE",
        "Miscalibration Area": ("Miscal.", "Area"),
        "picp": "PICP",
        "ence": "ENCE",
        "std_variation": "$C_{v}$",
        "aggregated_ence_cv": (
            "Aggr.",
            "ENCE",
            "$C_{v}$",
        ),  # ("ENCE+", "$0.1\cdot (1 - C_{v})$"),
    },
    "ЖК": {
        "model": "Модель",
        "future horizon months": (
            "Прогн.",
            "(мес.)",
        ),  # first name part will be on first line of table header, second part will be on second line
        "MAE": "MAE",
        "RMSE": "RMSE",
        "R2": "R2",
        "Root-mean-squared Calibration Error": "RMSCE",
        "Miscalibration Area": ("Miscal.", "Area"),
        "picp": "PICP",
        "ence": "ENCE",
        "std_variation": "$C_{v}$",
        "aggregated_ence_cv": (
            "Aggr.",
            "ENCE",
            "$C_{v}$",
        ),  # ("ENCE+", "$0.1\cdot (1 - C_{v})$"),
    },
}


columns_to_use = [
    "model",
    "future horizon months",
    "RMSE",
    "Root-mean-squared Calibration Error",
    "Miscalibration Area",
    "picp",
    "ence",
    "std_variation",
    # "aggregated_ence_cv",
]
NOT_USED_MODELS = []  # ["CatBoost", "ARIMA"]

CAPTION = f"""
Метрики качества регрессии и калибровки для суррогатной модели GPR на {DATASET_NAME} наборе данных.
"""

translate = {"ЖК": "jk", "ПК": "pk"}
LABEL = "\label{tab:surrogate_results_full_" + translate[DATASET_NAME] + "}\n"


def compute_best_values_for_each_horizon(data):
    best_rmse = data.groupby(by=["future horizon months"]).min()["RMSE"]
    best_rmsce = data.groupby(by=["future horizon months"]).min()[
        "Root-mean-squared Calibration Error"
    ]
    best_mical = data.groupby(by=["future horizon months"]).min()["Miscalibration Area"]
    best_picp = data.groupby(by=["future horizon months"]).max()["picp"]
    best_ence = data.groupby(by=["future horizon months"]).min()["ence"]
    best_std_variation = data.groupby(by=["future horizon months"]).max()[
        "std_variation"
    ]
    best_aggregated_ence_cv = data.groupby(by=["future horizon months"]).min()[
        "aggregated_ence_cv"
    ]
    best_values = {
        "RMSE": best_rmse,
        "Root-mean-squared Calibration Error": best_rmsce,
        "Miscalibration Area": best_mical,
        "picp": best_picp,
        "ence": best_ence,
        "std_variation": best_std_variation,
        "aggregated_ence_cv": best_aggregated_ence_cv,
    }
    return best_values


if __name__ == "__main__":
    data = pd.read_csv(METRIC_DATA_PATH)
    model_names = list(data["model"])
    good_rows = [
        np.all([(model_name not in name) for model_name in NOT_USED_MODELS])
        for name in model_names
    ]

    data = data.loc[good_rows]
    best_values = compute_best_values_for_each_horizon(data)
    result_latex_code = """"""
    result_latex_code += (
        "\\begin{longtable}[ht]{|l|"
        + "".join(["c|"] * (len(columns_to_use) - 1))
        + "}\n"
    )
    result_latex_code += "\t\\caption{" + CAPTION + "}\n"
    result_latex_code += LABEL
    result_latex_code += "\\\\\n"
    result_latex_code += "\\hline\n"

    # add header
    for column in columns_to_use:
        name = column_name_to_latex_name[DATASET_NAME][column]
        if isinstance(name, str):
            result_latex_code += name + " & "
        else:
            result_latex_code += name[0] + " & "
    # strip last &
    result_latex_code = result_latex_code[:-3]
    result_latex_code += " \\\\\n"
    # second header row
    for column in columns_to_use:
        name = column_name_to_latex_name[DATASET_NAME][column]
        if isinstance(name, str):
            result_latex_code += "~ & "
        else:
            result_latex_code += name[1] + " & "
    # strip last &
    result_latex_code = result_latex_code[:-3]
    result_latex_code += " \\\\\n"

    # third header row
    for column in columns_to_use:
        name = column_name_to_latex_name[DATASET_NAME][column]
        if isinstance(name, str) or len(name) == 2:
            result_latex_code += "~ & "
        else:
            result_latex_code += name[2] + " & "
    # strip last &
    result_latex_code = result_latex_code[:-3]
    result_latex_code += " \\\\\n"

    # add data

    old_horizon = None
    for index, row in data.iterrows():

        if not np.all(
            [(model_name not in row["model"]) for model_name in NOT_USED_MODELS]
        ):
            continue

        relevant_data = row[columns_to_use]

        relevant_data["future horizon months"] = int(
            relevant_data["future horizon months"]
        )
        # add hline

        if old_horizon != relevant_data["future horizon months"]:
            old_horizon = relevant_data["future horizon months"]
            result_latex_code += "\\hline\n"

        # if "t" in relevant_data["model"]:
        #     # split surrogate model name
        #     base_model = row["model"].split(",")[0]
        #     surr_model = ",".join(row["model"].split(",")[1:])[1:]
        #     result_latex_code += (
        #         " & ".join([base_model] + ["~"] * (len(columns_to_use) - 1)) + " \\\\\n"
        #     )

        #     relevant_data["model"] = surr_model
        relevant_data["model"] = relevant_data["model"].replace("GPRs, ", "")
        relevant_data["model"] = relevant_data["model"].replace("t1", "Sur I")
        relevant_data["model"] = relevant_data["model"].replace("t2", "Sur II")
        relevant_data["model"] = relevant_data["model"].replace("t3", "Sur III")
        row_values = []
        for key, value in dict(relevant_data).items():

            if key == "RMSE" and DATASET_NAME == "ПК":
                relevant_data[key] = round(value / 10 ** 3, 3)
            if (
                key in best_values
                and value
                == best_values[key][float(relevant_data["future horizon months"])]
            ):
                row_values.append("\\textbf{" + str(relevant_data[key]) + "}")

            else:
                if key == "future horizon months":
                    if relevant_data["model"] == "GPR":
                        s = (
                            "\\multirow{5}{*}{"
                            + str(relevant_data["future horizon months"])
                            + "}"
                        )
                        row_values.append(s)
                    else:
                        row_values.append(" ")
                else:
                    row_values.append(str(relevant_data[key]))

        result_latex_code += " & ".join(row_values) + " \\\\\n"

    result_latex_code += "\\hline\n"
    result_latex_code += "\\end{longtable}\n"

    with open("table.tex", "w") as fd:
        fd.write(result_latex_code)
