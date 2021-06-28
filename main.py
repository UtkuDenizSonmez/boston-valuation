from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)

features = data.drop(["INDUS", "AGE"], axis=1)  # Removed because of p-values lower than 0.05
log_prices = np.log(boston_dataset.target)  # Using Log prices because of lower skew.
target = pd.DataFrame(log_prices, columns=["PRICE"])  # Now it's 2 Dimentionals

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
NOX_IDX = 3
RM_IDX = 4
PTRATIO_IDX = 8

ZILLOW_MEDIAN_PRICE = 680.1
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

property_stats = np.ndarray(shape=(1, 11))
property_stats = features.mean().values.reshape(1, 11)  # Propert stats is now template to making the prediction

regr = LinearRegression().fit(features, target)
fitted_values = regr.predict(features)

MSE = mean_squared_error(target, fitted_values)
RMSE = np.sqrt(MSE)


def get_log_estimate(nr_of_rooms, students_per_classroom, next_to_river=False, high_confidence=True):
    # Configure Property
    property_stats[0][RM_IDX] = nr_of_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom

    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0

    # Make Prediction
    log_estimate = regr.predict(property_stats)[0][0]

    # Value Range
    if high_confidence:
        upper_bound = log_estimate + 2 * RMSE
        lower_bound = log_estimate - 2 * RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68

    return log_estimate, upper_bound, lower_bound, interval


def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    """
        Estimate price of a property in Boston.
        Keyword Arguments:
        rm -- number of rooms of the property.
        ptratio -- number of students per teacher in the classroom for the school in the area.
        chas -- True if the property close to the Chas River, False otherwise.
        large_range -- True for a 95% prediction interval, False for a 68% prediction interval.
    """

    log_est, upper, lower, conf = get_log_estimate(rm, ptratio, chas, large_range)

    if rm < 1 or ptratio < 0 or ptratio > 100:
        return "That is unrealistic. Try again."

    # Convert to today's dollars
    dollar_est = np.e ** log_est * 1000 * SCALE_FACTOR
    dollar_upper = np.e ** upper * 1000 * SCALE_FACTOR
    dollar_lower = np.e ** lower * 1000 * SCALE_FACTOR

    # Round the dollar values to nearest thousand
    rounded_est = np.round(dollar_est, -3)
    rounded_upper = np.round(dollar_upper, -3)
    rounded_lower = np.round(dollar_lower, -3)

    print("The Estimated Proper Value is: ", rounded_est)
    print(f"At {conf}% Confidence valudation range is")
    print(f"USD {rounded_lower} at the lower and {rounded_upper} at the high end.")
