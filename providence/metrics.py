# -*- coding: utf-8 -*-
from typing import List, Tuple, Union, Dict, types
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from lifelines.utils import concordance_index
from providence.distributions.weibull import compute_distribution_params

ArrayLike = Union[torch.Tensor, pd.Series, np.array]
Distribution = types.ModuleType


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    """
    Calculates mean square error of two vectors

    :param y_true: target vector
    :param y_pred: prediction vector

    :return MSE:
    """
    mse_ = round(np.average(np.square(y_true - y_pred), axis=0),2)
    return mse_


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    """
    Calculates symmetric mean absolute percentage error of two vectors

    :param y_true: target vector
    :param y_pred: prediction vector

    :return SMAPE:
    """
    s = {"pred": y_pred, "act": y_true}
    smape_df = pd.DataFrame(data=s)
    smape_ = (smape_df.pred-smape_df.act).abs()/(smape_df.act.abs()+smape_df.pred.abs())
    smape_ = np.nan_to_num(smape_, nan=0, posinf=0, neginf=0)
    return round(np.average(smape_),2)


def mfe(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    """
    Calculates mean forecast error of two vectors

    :param y_true: target vector
    :param y_pred: prediction vector

    :return MFE:
    """
    mfe_ = round(np.average((y_true - y_pred), axis=0),2)
    return mfe_


def smpe(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    """
    Calculates scaled mean percentage error of two vectors

    :param y_true: target vector
    :param y_pred: prediction vector

    :return SMPE:
    """
    s = {"pred": y_pred, "act": y_true}
    smpe_df = pd.DataFrame(data=s)
    smpe_ = (smpe_df.pred-smpe_df.act)/(smpe_df.act.abs()+smpe_df.pred.abs())
    smpe_ = np.nan_to_num(smpe_, nan=0, posinf=0, neginf=0)
    return round(np.average(smpe_),2)

def generate_metrics_table(y_true: ArrayLike, y_pred: ArrayLike) -> pd.DataFrame:

    d = {"MSE": [mse(y_true, y_pred)], "MFE": [mfe(y_true, y_pred)], "SMAPE": [smape(y_true, y_pred)], "SMPE": [smpe(y_true, y_pred)]}
    metrics = pd.DataFrame(data=d)

    return metrics


def generate_distribution_measures(
    distribution: Distribution, distribution_kwargs: Dict[str, torch.Tensor], tte_target: torch.Tensor
) -> pd.DataFrame:
    """
    Helper function to generate a dataframe of distribution results

    :param distribution: Module of providence.distributions
    :param distribution_kwargs: Dictionary of tensor arguments
    :param tte: torch.Tensor of [time_to_event, censor_bool]

    :return: Dataframe of distribution measures
    """
    measures_dict = distribution_kwargs

    measures_dict.update(
        {
            "tte": tte_target.numpy()[:, 0],
            "censor": tte_target.numpy()[:, 1],
            "mean": distribution.mean(**distribution_kwargs).numpy(),
            "median": distribution.median(**distribution_kwargs).numpy(),
            "mode": distribution.mode(**distribution_kwargs).numpy(),
        }
    )

    return pd.DataFrame.from_dict(measures_dict)


def output_per_device(
    model: torch.nn.Module, distribution: Distribution, fleet_object: Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]]
):
    """
    Function to calculate distribution measures for each device

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :fleet_object: Providence test dataset

    :return: Dataframe of each device's distribution measures
    """

    device_arr = []
    counter = 0

    for device in fleet_object:
        model.eval()
        counter = counter + 1
        alpha, beta = compute_distribution_params(model, device[0])
        ab = {"alpha": alpha, "beta": beta}

        device_df = generate_distribution_measures(distribution, ab, device[1])
        device_df["id"] = counter

        device_arr.append(device_df)

    device_output = pd.concat(device_arr)

    return device_output


def metrics_per_device(
    model: torch.nn.Module, distribution: Distribution, fleet_object: Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]]
):
    """
    Function to calculate performance metrics for each device

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset

    :return: Dataframe of performance metrics for each device in the fleet
    """

    metrics = []
    counter = 0

    for device in fleet_object:
        model.eval()
        counter = counter + 1

        alpha, beta = compute_distribution_params(model, device[0])
        ab = {"alpha": alpha, "beta": beta}

        device_df = generate_distribution_measures(distribution, ab, device[1])

        y_true, y_pred = device_df["tte"], device_df["mean"]
        device_metrics = generate_metrics_table(y_true, y_pred)
        device_metrics["id"] = counter

        metrics.append(device_metrics)

    metrics = pd.concat(metrics)

    return metrics


def fleet_metrics(
    model: torch.nn.Module, distribution: Distribution, fleet_object: Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]]
):
    """
    Function to calculate performance metrics for the fleet

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset

    :return: Dataframe of performance metrics for the fleet
    """
    fleet = output_per_device(model, distribution, fleet_object)
    y_true, y_pred = fleet["tte"], fleet["mean"]

    fleet_metrics = generate_metrics_table(y_true, y_pred)

    fleet_metrics["concordance_index"] = concordance_index(y_true, y_pred)

    return fleet_metrics

def window_predictions(
    model: torch.nn.Module, distribution: Distribution,
    fleet_object: Union[Dataset,List[Tuple[torch.Tensor, torch.Tensor]]],
    max_timestep: float, min_timestep: float
):
    """
    Function to generate predictions for each device at each timestep in a given window

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window

    return: dataframe
    """

    fleet = output_per_device(model, distribution, fleet_object)

    mask = fleet['tte'].between(min_timestep,max_timestep)
    window = fleet[mask]

    return window


def metrics_by_timestep(
        model: torch.nn.Module, distribution: Distribution,
        fleet_object: Union[Dataset,List[Tuple[torch.Tensor, torch.Tensor]]],
        max_timestep: float, min_timestep: float
):
    """
    Function to calculate MSE and MFE for all devices at each timestep

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window

    return: dataframe
    """
    window = window_predictions(model,distribution,fleet_object,max_timestep,min_timestep)

    metrics_window = []
    for timestep in window.tte.unique():
        df = window[(window['tte']==timestep)]
        mse_ = mse(df['tte'],df['mean'])
        mfe_ = mfe(df['tte'],df['mean'])
        output = pd.DataFrame({'tte':[timestep], 'mse':[mse_], 'mfe':[mfe_] })
        metrics_window.append(output)

    metrics_window = pd.concat(metrics_window)

    return metrics_window

def error_by_timestep(
        model: torch.nn.Module, distribution: Distribution,
        fleet_object: Union[Dataset,List[Tuple[torch.Tensor, torch.Tensor]]],
        max_timestep: float, min_timestep: float
):
    """
    Function to calculate error between actual TTE and mean, median, and mode
    of output distribution for each timestep.

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window

    return: dataframe
    """
    df = window_predictions(model,distribution,fleet_object,max_timestep,min_timestep)
    df['error_mean'] = df['tte'] - df['mean']
    df['error_median'] = df['tte'] - df['median']
    df['error_mode'] = df['tte'] - df['mode']
    df['mean_overshoot'] = np.where((df['error_mean']<0),True,False)
    df['median_overshoot'] = np.where((df['error_median']<0),True,False)
    df['mode_overshoot'] = np.where((df['error_mode']<0),True,False)
    return df

def percent_overshot_by_tte(
        model: torch.nn.Module, distribution: Distribution,
        fleet_object: Union[Dataset,List[Tuple[torch.Tensor, torch.Tensor]]],
        max_timestep: float, min_timestep: float
):
    """
    Function to calculate how many predictions have overshot the actual TTE at
    each timestep for mean, median, and mode

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window

    return: dataframe
    """
    error_by_ts = error_by_timestep(model,distribution,fleet_object,max_timestep,min_timestep)

    percent_overshot = []
    for i in error_by_ts.tte.unique():
        df = error_by_ts[(error_by_ts['tte']==i)]
        overshot_mean = (len(df[(df['mean_overshoot']==True)]))/(len(df))
        overshot_median = (len(df[(df['median_overshoot']==True)]))/(len(df))
        overshot_mode = (len(df[(df['mode_overshoot']==True)]))/(len(df))
        overshot_df = pd.DataFrame({"TTE":[i],"%_Overshot_Mean":[overshot_mean],"%_Overshot_Median":[overshot_median],"%_Overshot_Mode":[overshot_mode]})
        percent_overshot.append(overshot_df)

    overshot_by_tte = pd.concat(percent_overshot)

    return overshot_by_tte