# pyrefly: ignore [missing-import]
import lightkurve as lk
import numpy as np
import logging

# Configure logging to see the pipeline's progress
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_clean_lightcurve(tic_id: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Downloads and processes a TESS light curve for a given TIC ID.
    
    Args:
        tic_id (str): The TIC ID of the target (e.g., 'TIC 279741379').
        
    Returns:
        tuple: A tuple containing (time_array, flux_array) as 1D numpy arrays.
    """
    logger.info(f"Searching for TESS light curve for {tic_id}...")
    
    # 1. Ensure TIC format (if user just typed numbers, prepend 'TIC ')
    tic_id = str(tic_id).strip()
    if tic_id.isdigit():
        tic_id = f"TIC {tic_id}"
    elif not tic_id.upper().startswith("TIC"):
        tic_id = f"TIC {tic_id}"
        
    try:
        search_result = lk.search_lightcurve(tic_id, mission='TESS', author='SPOC')
        
        if len(search_result) == 0:
            logger.warning(f"No light curve found for {tic_id}.")
            return np.array([]), np.array([])
            
        logger.info(f"Found {len(search_result)} results. Downloading...")
        
        # 2. Download the data
        # We download ONLY the first available sector to prevent memory crashes and speed up downloads
        lc = search_result[0].download()
        if lc is None:
            logger.warning(f"Failed to download light curve for {tic_id}.")
            return np.array([]), np.array([])
            
    except Exception as e:
        logger.error(f"Network error communicating with NASA for {tic_id}: {e}")
        return np.array([]), np.array([])
    
    # 3. Process it: remove NaNs, remove outliers, and flatten
    logger.info("Cleaning light curve (removing NaNs, outliers, and flattening)...")
    clean_lc = lc.remove_nans().remove_outliers().flatten(window_length=101)
    
    # 4. Extract time and flux values and return 1D numpy arrays
    time_1d = clean_lc.time.value
    flux_1d = clean_lc.flux.value
    
    logger.info(f"Successfully processed {tic_id}. Returning arrays of shape {flux_1d.shape}.")
    return time_1d, flux_1d

if __name__ == "__main__":
    # --- Testing Block ---
    # For testing, we hardcode TIC 279741379 (a known exoplanet) and fold it at period=4.09.
    
    test_tic = "TIC 279741379"
    test_period = 4.09
    
    print("="*50)
    print(f"Running Data Pipeline Test for {test_tic}")
    print("="*50)
    
    # We replicate the pipeline steps here to retain the lightkurve object so we can use its .fold() method.
    # (Since `get_clean_lightcurve` strictly returns a 1D numpy array, it drops the time axis needed for folding).
    
    search_result = lk.search_lightcurve(test_tic, mission='TESS', author='SPOC')
    
    if len(search_result) > 0:
        print(f"Downloading data for {test_tic}...")
        lc = search_result.download_all().stitch()
        
        print("Cleaning and flattening...")
        clean_lc = lc.remove_nans().remove_outliers().flatten(window_length=101)
        
        print(f"Folding light curve at period = {test_period} days...")
        folded_lc = clean_lc.fold(period=test_period)
        
        # Extract the flux values and return a 1D numpy.ndarray
        final_flux_array = folded_lc.flux.value
        
        print("\n--- Test Results ---")
        print(f"Successfully extracted folded 1D flux array!")
        print(f"Type:  {type(final_flux_array)}")
        print(f"Shape: {final_flux_array.shape}")
        print(f"First 10 flux values:\n{final_flux_array[:10]}")
        print("="*50)
    else:
        print(f"Could not find data for test TIC: {test_tic}")
