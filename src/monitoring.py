from datetime import datetime, timedelta

import pandas as pd

import src.config as config
from src.feature_store_api import get_feature_store, get_feature_group


def load_predictions_and_actual_values_from_store(
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    """Fetches model predictions and actuals values from
    `from_date` to `to_date` from the Feature Store and returns a dataframe

    Args:
        from_date (datetime): min datetime for which we want predictions and
        actual values

        to_date (datetime): max datetime for which we want predictions and
        actual values

    Returns:
        pd.DataFrame: 4 columns
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
            - `rides`
    """
    # 2 feature groups we need to merge
    predictions_fg = get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTIONS)
    actuals_fg = get_feature_group(name=config.FEATURE_GROUP_NAME)

    # query to join the 2 features groups by `pickup_hour` and `pickup_location_id`
    query = predictions_fg.select_all() \
        .join(actuals_fg.select_all(), on=['pickup_hour', 'pickup_location_id']) \
        .filter(predictions_fg.pickup_hour >= from_date) \
        .filter(predictions_fg.pickup_hour <= to_date)
    
    # create the feature view `config.FEATURE_VIEW_MONITORING` if it does not
    # exist yet
    feature_store = get_feature_store()
    try:
        # create feature view as it does not exist yet
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_MONITORING,
            version=1,
            query=query
        )
    except:
        print('Feature view already existed. Skip creation.')

    # feature view
    monitoring_fv = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_MONITORING,
        version=1
    )

    # fetch data form the feature view
    # fetch predicted and actual values for the last 30 days
    monitoring_df = monitoring_fv.get_batch_data(
        start_time=from_date - timedelta(days=7),
        end_time=to_date + timedelta(days=7)
    )
    monitoring_df = monitoring_df[monitoring_df.pickup_hour.between(from_date, to_date)]

    return monitoring_df
