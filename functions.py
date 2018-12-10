# Your code here - remember to use markdown cells for comments as well!
import math
import pandas as pd
import numpy as np

from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols

import matplotlib.pyplot as plt

import folium
from folium.plugins import HeatMap

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    """
    Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS

    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features

    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """

    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True

            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        # null if pvalues is empty
        worst_pval = pvalues.max()

        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)

            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break

    return included


def display_heatmap(data):
    """
    Display a heatmap from a given dataset

    :param data: dataset
    :return: g (graph to display)
    """

    # Set the style of the visualization
    # sns.set(style = "white")
    sns.set_style("white")

    # Create a covariance matrix
    corr = data.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = None

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 12))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, sep=20, n=9, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap, mask=mask, square=True)

    return g


def display_jointplot(data, columns):
    """
    Display seaborn jointplot on given dataset and feature list

    :param data: dataset
    :param columns: feature list
    :return: g
    """

    sns.set_style('whitegrid')

    for column in columns:
        g = sns.jointplot(x=column, y="price", data=data, dropna=True,
                          kind='reg', joint_kws={'line_kws': {'color': 'red'}})

    return g


def display_plot(data, vars, target, plot_type='box'):
    """
    Generates a seaborn boxplot (default) or scatterplot

    :param data: dataset
    :param vars: feature list
    :param target: feature name
    :param plot_type: box (default), scatter, rel
    :return: g
    """

    # pick one dimension
    ncol = 3
    # make sure enough subplots
    nrow = math.floor((len(vars) + ncol - 1) / ncol)
    # create the axes
    fig, axarr = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, 20))

    # go over a linear list of data
    for i in range(len(vars)):
        # compute an appropriate index (1d or 2d)
        ix = np.unravel_index(i, axarr.shape)

        feature_name = vars[i]

        if plot_type == 'box':
            g = sns.boxplot(y=feature_name, x=target, data=data, width=0.8,
                            orient='h', showmeans=True, fliersize=3, ax=axarr[ix])

        # elif plot_type == 'scatter':
        else:
            g = sns.scatterplot(x=feature_name, y=target, data=data, ax=axarr[ix])

        # else:
        #     col_name = vars[i]
        #     g = sns.relplot(x=feature_name, y=target, hue=target, col=col_name,
        #                     size=target, sizes=(5, 500), col_wrap=3, data=data)

    return g


def map_feature_by_zipcode(zipcode_data, col):
    """
    Generates a folium map of Seattle
    :param zipcode_data: zipcode dataset
    :param col: feature to display
    :return: m
    """

    # read updated geo data
    king_geo = "cleaned_geodata.json"

    # Initialize Folium Map with Seattle latitude and longitude
    m = folium.Map(location=[47.35, -121.9], zoom_start=9,
                   detect_retina=True, control_scale=False)
    # tiles='stamentoner')

    # Create choropleth map
    m.choropleth(
        geo_data=king_geo,
        name='choropleth',
        data=zipcode_data,
        # col: feature of interest
        columns=['zipcode', col],
        key_on='feature.properties.ZIPCODE',
        fill_color='OrRd',
        fill_opacity=0.9,
        line_opacity=0.2,
        legend_name='house ' + col
    )

    folium.LayerControl().add_to(m)

    # Save map based on feature of interest
    m.save(col + '.html')

    return m


def measure_strength(data, feature_list, target):
    """
    Calculate a Pearson correlation coefficient and the p-value to test for non-correlation.

    :param data: dataset
    :param feature_list: feature list
    :param target: feature name
    :return:
    """

    print("Pearson correlation coefficient R and p-value \n\n")

    for k, v in enumerate(feature_list):
        r, p = stats.pearsonr(data[v], data[target])
        print("{0} <=> {1}\t\tR = {2} \t\t p = {3}".format(target, v, r, p))


def heatmap_features_by_loc(data, feature):
    """
    Generates a heatmap based on lat, long and a feature

    :param data: dataset
    :param feature: feature name
    :return:
    """
    max_value = data[feature].max()

    lat = np.array(data.lat, dtype=pd.Series)
    lon = np.array(data.long, dtype=pd.Series)
    mag = np.array(data[feature], dtype=pd.Series) / max_value

    d = np.dstack((lat, lon, mag))[0]
    heatmap_data = [i for i in d.tolist()]

    hmap = folium.Map(location=[47.55, -122.0], zoom_start=10, tiles='stamentoner')

    hm_wide = HeatMap(heatmap_data,
                      min_opacity=0.7,
                      max_val=max_value,
                      radius=1, blur=1,
                      max_zoom=1,
                      )

    hmap.add_child(hm_wide)

    return hmap
