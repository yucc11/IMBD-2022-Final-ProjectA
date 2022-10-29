"""
EDA utilities.
Author: JiaWei Jiang
"""
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import display

colors = sns.color_palette("Set2")


def summarize(
    df: pd.DataFrame,
    file_name: Optional[str] = None,
    n_rows_to_display: Optional[int] = 5,
) -> None:
    """Summarize DataFrame.

    Parameters:
        df: input data
        file_name: name of the input file
        n_rows_to_display: number of rows to display

    Return:
        None
    """
    file_name = "Data" if file_name is None else file_name

    # Derive NaN ratio for each column
    nan_ratio = pd.isna(df).sum() / len(df) * 100
    nan_ratio.sort_values(ascending=False, inplace=True)
    nan_ratio = nan_ratio.to_frame(name="NaN Ratio").T

    # Derive zero ratio for each column
    zero_ratio = (df == 0).sum() / len(df) * 100
    zero_ratio.sort_values(ascending=False, inplace=True)
    zero_ratio = zero_ratio.to_frame(name="Zero Ratio").T

    # Print out summarized information
    print(f"=====Summary of {file_name}=====")
    display(df.head(n_rows_to_display))
    print(f"Shape: {df.shape}")
    print("NaN ratio:")
    display(nan_ratio)
    print("Zero ratio:")
    display(zero_ratio)


def plot_univar_dist(data: Union[pd.Series, np.ndarray], feature: str, bins: int = 250) -> None:
    """Plot univariate distribution.

    Parameters:
        data: univariate data to plot
        feature: feature name of the data
        bins: number of bins

    Return:
        None
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=data, bins=bins, kde=True, palette=colors, ax=ax)
    ax.axvline(x=data.mean(), color="orange", linestyle="dotted", linewidth=1.5, label="Mean")
    ax.axvline(
        x=data.median(),
        color="green",
        linestyle="dotted",
        linewidth=1.5,
        label="Median",
    )
    ax.axvline(
        x=data.mode().values[0],
        color="red",
        linestyle="dotted",
        linewidth=1.5,
        label="Mode",
    )
    ax.set_title(
        f"{feature.upper()} Distibution\n"
        f"Min {round(data.min(), 2)} | "
        f"Max {round(data.max(), 2)} | "
        f"Skewness {round(data.skew(), 2)} | "
        f"Kurtosis {round(data.kurtosis(), 2)}"
    )
    ax.set_xlabel(f"{feature}")
    ax.set_ylabel("Bin Count")
    ax.legend()
    plt.show()


def plot_series(
    df: pd.DataFrame,
    features: List[str],
    title: Optional[str] = None,
    x_axis: Optional[str] = None,
) -> None:
    """Plot series on the same figure.

    With multiple features plotted on the same figure, analyzers can
    observe if there's any synchronous behavior in each feature pair.

    Parameters:
        df: input data
        features: list of feature names
        title: title of the figure
        x_axis: name of the column acting as x axis

    Return:
        None
    """
    x = np.arange(len(df)) if x_axis is None else df[x_axis]

    fig = go.Figure()
    for f in features:
        fig.add_trace(go.Scatter(x=x, y=df[f], mode="lines", name=f))
    if title is not None:
        fig.update_layout(title=title)
    fig.show()


def plot_bivar(
    data: Union[pd.Series, np.ndarray],
    features: Optional[List[str]] = ["0", "1"],
) -> None:
    """Plot bivariate distribution with regression line fitted.

    Parameters:
        data: bivariate data to plot
        features: list of feature names

    Return:
        None
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    f1, f2 = features[0], features[1]
    corr = data[[f1, f2]].corr().iloc[0, 1]

    ax = sns.jointplot(
        x=data[f1],
        y=data[f2],
        kind="reg",
        height=6,
        marginal_ticks=True,
        joint_kws={"line_kws": {"color": "orange"}},
    )
    ax.fig.suptitle(f"{f1} versus {f2}, Corr={corr:.2}")
    ax.ax_joint.set_xlabel(f1)
    ax.ax_joint.set_ylabel(f2)
    plt.tight_layout()


def plot_nan_ratios(df: pd.DataFrame, xlabel: str = "Feature") -> None:
    """Plot NaN ratio bar plot ranked by ratio values.

    Parameters:
        df: input data
        xlabel: label on x-axis

    Return:
        None
    """
    nan_ratios = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    nan_ratios = nan_ratios[nan_ratios != 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=nan_ratios.index, y=nan_ratios.values, palette=colors, ax=ax)
    ax.set_title(f"NaN Ratios of Different {xlabel}s")
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=90, labelsize="small")
    ax.set_ylabel("NaN Ratio")
    plt.show()


def plot_corr_heatmap(
        corr: pd.DataFrame,
        process_name: str=""
    ):
    """Plot correlation heatmap
    
    Parameters
    ----------
        corr: DataFrame 
            table of correlation coefficients
        
    Return:
        None
    """
    if process_name == "":
        title = "Correlation between 6 target variables"
    else:
       title = f"Correlation between {process_name} processes and 6 target variables"

    fig = go.Figure(data=go.Heatmap(
                        z=corr,
                        x=corr.index.values,
                        y=corr.index.values,
                        colorscale='Viridis'
                        ))
    fig.update_layout(
        height=800, 
        width=800,
        title=title)
    fig.show()


def desc_correlation(
        corr: pd.DataFrame,
        cols_1: List=[str], 
        cols_2: List=[str],
        show_all: bool=False
        ):
    """Describe the strength of correlation coefficients
    
    Parameters
    ----------
        corr: DataFrame 
            table of correlation coefficients
        cols_1: List=[str]
            list of sub-process names
        cols_2: List=[str]
            list of sub-process names
        show_all: Boolean
            Show all relationships. Display strong and moderate relationship by default.
        
    Return:
        pd.DataFrame
    """
    strong, mod, weak, no = [], [], [], []
    for col1 in cols_1:
        for col2 in cols_2:
            if col1 == col2:
                continue
            c = corr.loc[col1, col2]
            if (c >= 0.75) or (c <= -0.75):
                strong.append([col1, col2, c])
            elif (c < 0.75 and c >= 0.5) or (c > -0.75 and c <= -0.5):
                mod.append([col1, col2, c])
            elif (c < 0.5 and c >= 0.25) or (c > -0.5 and c <= -0.25):
                weak.append([col1, col2, c])
            elif (c < 0.25 and c >= 0) or (c > -0.25 and c <= 0):
                no.append([col1, col2, c])

    print("Strong relationship:")
    strong = pd.DataFrame(strong, columns=["subprocess_1", "subprocess_2", "correlation"])
    display(strong)
    print("Moderate relationship:")
    mod = pd.DataFrame(mod, columns=["subprocess_1", "subprocess_2", "correlation"])
    display(mod)

    if show_all:
        print("Weak relationship:")
        display(pd.DataFrame(weak, columns=["subprocess_1", "subprocess_2", "correlation"]))
        print("No relationship:")
        display(pd.DataFrame(no, columns=["subprocess_1", "subprocess_2", "correlation"]))
    
    return strong, mod

