# model/post_processor.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter

class PostProcessor:
    """
    A class to provide various post-processing methods for cleaning
    and refining extracted data series.
    """

    @staticmethod
    def remove_outliers_local_regression(points, window_fraction=0.15, threshold=3.0):
        """
        Removes outliers using a LOESS-inspired local regression method.
        For each point, it performs a weighted linear regression on its local
        neighborhood and removes points that deviate significantly. This is more
        robust than a single global polynomial fit.
        """
        if len(points) < 10:
            print("Warning: Not enough points for local regression outlier removal. Skipping.")
            return points

        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])

        # Sort by x to ensure windows are contiguous along the axis
        sort_indices = np.argsort(x)
        x_sorted, y_sorted = x[sort_indices], y[sort_indices]
        original_points_sorted = [points[i] for i in sort_indices]

        window_size = max(5, int(len(x) * window_fraction))
        if window_size >= len(x):
            window_size = len(x) - 1

        residuals = np.zeros(len(x))

        for i in range(len(x_sorted)):
            # Define the window of points around the current point
            start = max(0, i - window_size // 2)
            end = min(len(x_sorted), i + window_size // 2 + 1)
            x_window, y_window = x_sorted[start:end], y_sorted[start:end]

            if len(x_window) < 2: continue

            # Calculate weights (tricube weight function)
            distances = np.abs(x_window - x_sorted[i])
            max_dist = np.max(distances)
            weights = (1 - (distances / max_dist) ** 3) ** 3 if max_dist > 0 else np.ones_like(distances)

            # Perform weighted linear regression on the window
            try:
                X_window = np.vstack([x_window, np.ones(len(x_window))]).T
                XTWX_inv = np.linalg.inv(X_window.T @ (weights[:, np.newaxis] * X_window))
                coeffs = XTWX_inv @ (X_window.T @ (weights * y_window))
                predicted_y = coeffs[0] * x_sorted[i] + coeffs[1]
                residuals[i] = y_sorted[i] - predicted_y
            except np.linalg.LinAlgError:
                continue

        # Use Median Absolute Deviation (MAD) for a robust measure of spread
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        if mad == 0: mad = np.std(residuals) # Fallback for zero MAD
        if mad == 0: return original_points_sorted # No deviation

        modified_z_scores = 0.6745 * (residuals - median_residual) / mad
        inliers_mask = np.abs(modified_z_scores) < threshold
        inlier_points = [original_points_sorted[i] for i, is_inlier in enumerate(inliers_mask) if is_inlier]

        print(f"Local outlier removal: {len(points) - len(inlier_points)} points removed.")
        return inlier_points

    @staticmethod
    def resample_series(points, num_points=100, method='linear'):
        if len(points) < 2: return points

        # --- FIX: Handle duplicate x-values to prevent interpolation errors ---
        df = pd.DataFrame(points)
        df_unique = df.groupby('x', as_index=False).mean().sort_values('x')

        if len(df_unique) < 2:
            return points  # Not enough unique points to resample

        x_sorted = df_unique['x'].to_numpy()
        y_sorted = df_unique['y'].to_numpy()

        if method == 'cubic_spline' and len(x_sorted) > 3:
            interp_func = CubicSpline(x_sorted, y_sorted)
        else:
            interp_func = interp1d(x_sorted, y_sorted, bounds_error=False, fill_value="extrapolate")
        x_new = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
        y_new = interp_func(x_new)
        resampled_points = [{'x': px, 'y': py} for px, py in zip(x_new, y_new)]
        return resampled_points

    @staticmethod
    def process(series_data, params, override_points=None):
        """
        Main entry point for post-processing a list of data series.
        Applies a sequence of cleaning and smoothing operations based on params.
        If override_points is provided, it will be used instead of the series' own points.
        """
        processed_series_list = []

        for series in series_data:
            # Start with the original points from the series
            points = series['data_points']

            # 1. Outlier Removal (if enabled AND not in manual editing mode)
            if params.get('outlier_removal_enabled', False) and not params.get('manual_editing_enabled', False):
                threshold = params.get('outlier_threshold', 3.0)
                window_fraction = params.get('outlier_window_fraction', 0.15)
                points = PostProcessor.remove_outliers_local_regression(points, window_fraction, threshold)

            # 2. Resampling Logic
            if params.get('manual_editing_enabled', False):
                # In manual mode, the override_points are the reference points.
                # We resample *from* these points.
                if override_points is not None:
                    num_points = params.get('manual_final_points', 100)
                    method = params.get('manual_interp_method', 'linear')
                    points = PostProcessor.resample_series(override_points, num_points, method)
                else:
                    # If no override points, just use the current points (likely from outlier removal or original)
                    # This case shouldn't really happen if the UI is working correctly, but it's a safe fallback.
                    pass # points remain as they are
            elif params.get('auto_resampling_enabled', False):
                # In auto mode, we resample the (potentially outlier-filtered) points.
                num_points = params.get('auto_resample_points', 100)
                method = params.get('auto_resample_method', 'linear')
                points = PostProcessor.resample_series(points, num_points, method)

            new_series = series.copy()
            new_series['data_points'] = points
            processed_series_list.append(new_series)

        return processed_series_list