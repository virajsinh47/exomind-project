import numpy as np
from astropy.timeseries import BoxLeastSquares

def calculate_transit_stats(time_array: np.ndarray, flux_array: np.ndarray) -> dict:
    """
    Calculates transit statistics using Box Least Squares.
    Returns the orbital period, transit depth, and SNR.
    """
    # Initialize the BLS model
    model = BoxLeastSquares(time_array, flux_array)
    
    # We search for periods between 0.5 and 20 days, and transit durations between 0.01 and 0.2 days
    durations = np.linspace(0.01, 0.2, 10)
    results = model.autopower(durations, minimum_period=0.5, maximum_period=20.0)
    
    # Extract the best fit parameters
    best_idx = np.argmax(results.power)
    best_period = results.period[best_idx]
    best_duration = results.duration[best_idx]
    best_t0 = results.transit_time[best_idx]
    
    # Create a boolean mask for points that are in-transit
    # Phase ranges from -0.5 * period to 0.5 * period, centered at 0 during transit
    phase = (time_array - best_t0 + 0.5 * best_period) % best_period - 0.5 * best_period
    in_transit = np.abs(phase) < (0.5 * best_duration)
    
    # Calculate transit depth (1.0 - median dip flux)
    if np.any(in_transit):
        median_dip_flux = np.median(flux_array[in_transit])
    else:
        median_dip_flux = 1.0
        
    transit_depth = 1.0 - median_dip_flux
    
    # Calculate SNR (Depth / out-of-transit standard deviation)
    out_of_transit = ~in_transit
    if np.any(out_of_transit):
        out_transit_std = np.std(flux_array[out_of_transit])
    else:
        out_transit_std = np.inf
        
    if out_transit_std > 0:
        snr = transit_depth / out_transit_std
    else:
        snr = 0.0
        
    return {
        "orbital_period_days": float(best_period),
        "transit_duration_days": float(best_duration),
        "transit_depth": float(transit_depth),
        "snr": float(snr)
    }

if __name__ == "__main__":
    # Generate a fake dataset: 20 days of observations
    np.random.seed(42)
    time_fake = np.linspace(0, 20, 2000)
    
    # Base flux is a sine wave with some noise to represent stellar variability
    # Mean of 1.0
    flux_fake = 1.0 + 0.005 * np.sin(2 * np.pi * time_fake / 3.0) + np.random.normal(0, 0.001, len(time_fake))
    
    # Inject a small dip (transit) starting at day 4.0, with a period of 4.0 days
    true_period = 4.0
    true_t0 = 4.0
    true_duration = 0.1
    true_depth = 0.02
    
    # Add transits to the light curve
    for i in range(10):
        t_transit = true_t0 + i * true_period
        in_dip = np.abs(time_fake - t_transit) < (true_duration / 2)
        flux_fake[in_dip] -= true_depth
        
    print("Testing calculate_transit_stats with a simulated light curve...")
    print(f"Injected Period: {true_period} days")
    print(f"Injected Depth:  {true_depth}")
    print("-" * 40)
    
    # Run analytics
    stats = calculate_transit_stats(time_fake, flux_fake)
    
    print("Recovered Statistics:")
    print(f"  Orbital Period: {stats['orbital_period_days']:.4f} days")
    print(f"  Transit Depth:  {stats['transit_depth']:.4f}")
    print(f"  SNR:            {stats['snr']:.2f}")
