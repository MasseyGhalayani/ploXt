# model/post_processor.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
class PostProcessor:
    """
    A class to provide various post-processing methods for cleaning
    and refining extracted data series.
    """

    @staticmethod
    def remove_outliers_local_regression(points, window_fraction=0.15, threshold=3.0, x_scale='linear', y_scale='linear'):
        """
        Removes outliers using a LOESS-inspired local regression method.
        For each point, it performs a weighted linear regression on its local
        neighborhood and removes points that deviate significantly. This is more
        robust than a single global polynomial fit.
        --- NEW: Now scale-aware. It performs the regression in log-space for
        logarithmic axes to correctly model the data's behavior.
        """
        # --- NEW: Filter for positive values required by log scales first ---
        original_len = len(points)
        if y_scale == 'log':
            points = [p for p in points if p.get('y', 0) > 0]
        if x_scale == 'log':
            points = [p for p in points if p.get('x', 0) > 0]

        if len(points) != original_len:
            print(f"Outlier detection: Removed {original_len - len(points)} non-positive points for log scale processing.")

        if len(points) < 10:
            print("Warning: Not enough valid points for local regression outlier removal. Skipping.")
            return points

        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])

        # --- NEW: Transform data to the appropriate space for regression ---
        # This makes the linear regression work for linear, semilog, and log-log plots.
        x_reg = np.log10(x) if x_scale == 'log' else x
        y_reg = np.log10(y) if y_scale == 'log' else y

        # Sort by the independent variable (x_reg) to ensure windows are contiguous
        sort_indices = np.argsort(x_reg)
        x_sorted, y_sorted = x_reg[sort_indices], y_reg[sort_indices]
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
    def remove_outliers_ransac(points, x_scale='linear', y_scale='linear'):
        """
        Removes outliers using the RANSAC algorithm. It fits a robust polynomial
        model and removes points that are not considered inliers.
        --- REVERTED: Now uses a robust polynomial fit instead of a random forest. ---
        """
        if not SKLEARN_AVAILABLE:
            print("Warning: scikit-learn not found. RANSAC outlier removal is unavailable. Skipping.")
            return points
            
        original_len = len(points)
        if y_scale == 'log':
            points = [p for p in points if p.get('y', 0) > 0]
        if x_scale == 'log':
            points = [p for p in points if p.get('x', 0) > 0]

        if len(points) != original_len:
            print(f"RANSAC: Removed {original_len - len(points)} non-positive points for log scale processing.")

        if len(points) < 10:
            print("Warning: Not enough valid points for RANSAC. Skipping.")
            return points

        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])

        # Transform data to the appropriate space for regression
        x_reg = np.log10(x) if x_scale == 'log' else x
        y_reg = np.log10(y) if y_scale == 'log' else y

        x_reg_reshaped = x_reg.reshape(-1, 1)

        # --- FIX: Manually calculate a robust threshold based on residuals ---
        # The default `residual_threshold=None` in scikit-learn calculates the MAD
        # on the target values `y`, which is not suitable for data with a trend.
        # It results in a threshold that is too large and fails to remove outliers.
        # We will calculate the MAD on the *residuals* of a preliminary fit, which
        # is a much better estimate of the noise level around the trend.
        try:
            # 1. Fit a non-robust model to get the general trend.
            p_coeffs = np.polyfit(x_reg, y_reg, 2)
            p = np.poly1d(p_coeffs)
            # 2. Calculate the residuals (errors) from this trendline.
            residuals = y_reg - p(x_reg)
            # 3. Calculate the MAD of the residuals. This is a robust measure of noise.
            residual_mad = np.median(np.abs(residuals - np.median(residuals)))
            # 4. Set the threshold. A threshold of 3.0 * MAD is a common choice.
            robust_threshold = 3.0 * residual_mad
        except (np.linalg.LinAlgError, ValueError):
            # Fallback if the initial fit fails
            robust_threshold = np.median(np.abs(y_reg - np.median(y_reg)))


        model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=robust_threshold,
                                                                    random_state=42))

        try:
            model.fit(x_reg_reshaped, y_reg)
            # When using a pipeline, the RANSAC estimator instance is named 'ransacregressor'
            inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
        except Exception as e:
            print(f"RANSAC fitting failed: {e}")
            import traceback
            traceback.print_exc()
            inlier_mask = np.zeros(len(points), dtype=bool) # Mark all as outliers if fitting fails

        if not np.any(inlier_mask): # If no inliers found, return original points
            # RANSAC can fail if it can't find a consensus. Fallback gracefully.
            print("Warning: RANSAC failed to find a valid consensus set. Returning original points.")
            return points

        inlier_points = [p for (p, is_inlier) in zip(points, inlier_mask) if is_inlier]
        print(f"RANSAC outlier removal: {len(points) - len(inlier_points)} points removed.")
        return inlier_points

    @staticmethod
    def _apply_savgol_filter(x, y, num_points):
        """Helper to apply Savitzky-Golay filter."""
        if not SCIPY_AVAILABLE:
            print("Warning: SciPy not found. Savitzky-Golay filter is unavailable.")
            return y

        # Window length must be odd and less than the number of points
        window_length = min(len(x), max(5, int(num_points * 0.1)))
        if window_length % 2 == 0:
            window_length -= 1
        return savgol_filter(y, window_length, polyorder=2) if window_length > 2 else y

    @staticmethod
    def resample_series(points, num_points=100, method='linear', x_scale='linear', y_scale='linear'):
        """
        Resamples a series to a specified number of points.
        --- NEW: Now handles logarithmic scales for both axes. ---
        """
        if len(points) < 2:
            return points

        # Use pandas for easier data manipulation
        df = pd.DataFrame(points)

        # --- NEW: Filter out non-positive values which are invalid for log scales ---
        if y_scale == 'log':
            df = df[df['y'] > 0]
        if x_scale == 'log':
            # Also filter x for log scale and ensure min > 0 for geomspace
            df = df[df['x'] > 0]

        if len(df) < 2:
            print("Warning: Not enough valid points for resampling after filtering for log scale. Skipping.")
            return df.to_dict('records')

        # Handle duplicate x-values by averaging them to prevent interpolation errors
        df_unique = df.groupby('x', as_index=False).mean().sort_values('x')

        if len(df_unique) < 2:
            return df_unique.to_dict('records')  # Not enough unique points to resample

        x_sorted = df_unique['x'].to_numpy()
        y_for_interp = df_unique['y'].to_numpy()

        # --- NEW: Transform y-data to log space if y-axis is logarithmic ---
        if y_scale == 'log':
            y_for_interp = np.log10(y_for_interp)

        # --- NEW: Generate new x-points in either linear or log space ---
        x_new = np.geomspace(x_sorted.min(), x_sorted.max(), num_points) if x_scale == 'log' else np.linspace(x_sorted.min(), x_sorted.max(), num_points)

        if method == 'savitzky-golay':
            if not SCIPY_AVAILABLE: return points
            y_smoothed = PostProcessor._apply_savgol_filter(x_sorted, y_for_interp, len(x_sorted))
            interp_func = interp1d(x_sorted, y_smoothed, bounds_error=False, fill_value="extrapolate")
            y_new = interp_func(x_new)
        elif method == 'cubic_spline' and len(x_sorted) > 3:
            interp_func = CubicSpline(x_sorted, y_for_interp)
            y_new = interp_func(x_new)
        else:
            # Default to linear
            interp_func = interp1d(x_sorted, y_for_interp, bounds_error=False, fill_value="extrapolate")
            y_new = interp_func(x_new)

        # --- NEW: Transform y-data back from log space if needed ---
        if y_scale == 'log':
            y_new = 10**y_new

        resampled_points = [{'x': px, 'y': py} for px, py in zip(x_new, y_new)]
        return resampled_points

    @staticmethod
    def process(series_data, params, override_points=None, x_scale='linear', y_scale='linear', outlier_method='local_regression'):
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
                threshold = params.get('outlier_threshold', 3.0) # Used by both methods
                if outlier_method == 'ransac':
                    points = PostProcessor.remove_outliers_ransac(points, x_scale, y_scale)
                else: # Default to local regression
                    window_fraction = params.get('outlier_window_fraction', 0.15)
                    points = PostProcessor.remove_outliers_local_regression(points, window_fraction, threshold, x_scale, y_scale)

            # 2. Resampling Logic
            if params.get('manual_editing_enabled', False):
                # In manual mode, the override_points are the reference points.
                # We resample *from* these points.
                if override_points is not None:
                    num_points = params.get('manual_final_points', 100)
                    method = params.get('manual_interp_method', 'linear')
                    points = PostProcessor.resample_series(override_points, num_points, method, x_scale, y_scale)
                else:
                    # If no override points, just use the current points (likely from outlier removal or original)
                    # This case shouldn't really happen if the UI is working correctly, but it's a safe fallback.
                    pass # points remain as they are
            elif params.get('auto_resampling_enabled', False):
                # In auto mode, we resample the (potentially outlier-filtered) points.
                num_points = params.get('auto_resample_points', 100)
                method = params.get('auto_resample_method', 'linear')
                points = PostProcessor.resample_series(points, num_points, method, x_scale, y_scale)

            new_series = series.copy()
            new_series['data_points'] = points
            processed_series_list.append(new_series)

        return processed_series_list