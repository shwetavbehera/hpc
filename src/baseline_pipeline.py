"""
DEMREG Baseline Pipeline
========================
This script provides a baseline implementation for computing Differential Emission Measure (DEM)
from AIA solar observations.

Author: HPC Assignment - Baseline Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import sys
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dn2dem_pos import dn2dem_pos


def load_aia_data(data_dir):
    """
    Load AIA FITS files and extract data
    
    Returns:
        dict: Dictionary with wavelengths as keys and (data, header) as values
    """
    wavelengths = ['94', '131', '171', '193', '211', '304', '335']
    aia_data = {}
    
    print("Loading AIA FITS files...")
    for wl in wavelengths:
        # Find file matching wavelength
        files = list(Path(data_dir).glob(f'*{wl}A*.fits'))
        if files:
            filepath = files[0]
            print(f"  Loading {wl}Å: {filepath.name}")
            with fits.open(filepath) as hdul:
                data = hdul[1].data  # AIA data is usually in extension 1
                header = hdul[1].header
                aia_data[wl] = {'data': data, 'header': header}
        else:
            print(f"  Warning: No file found for {wl}Å")
    
    return aia_data


def prepare_dn_data(aia_data, wavelengths=['94', '131', '171', '193', '211', '335']):
    """
    Prepare DN data arrays for DEM calculation
    
    Args:
        aia_data: Dictionary of AIA data
        wavelengths: List of wavelengths to use (6 channels for DEM)
    
    Returns:
        dn_array: DN counts array
        edn_array: Error on DN counts
        exp_times: Exposure times for normalization
    """
    # Get image shape from first available wavelength
    first_wl = next(iter(aia_data.values()))
    img_shape = first_wl['data'].shape
    
    # Initialize arrays
    nf = len(wavelengths)
    dn_array = np.zeros((*img_shape, nf))
    edn_array = np.zeros((*img_shape, nf))
    exp_times = np.zeros(nf)
    
    print("\nPreparing DN data...")
    for i, wl in enumerate(wavelengths):
        if wl in aia_data:
            data = aia_data[wl]['data']
            header = aia_data[wl]['header']
            
            # Get exposure time
            exp_time = header.get('EXPTIME', 1.0)
            exp_times[i] = exp_time
            
            # Normalize by exposure time to get DN/s/px
            dn_array[:, :, i] = data / exp_time
            
            # Estimate error as sqrt(counts) / exp_time (Poisson statistics)
            # Add small value to avoid sqrt(0)
            edn_array[:, :, i] = np.sqrt(np.abs(data) + 1) / exp_time
            
            print(f"  {wl}Å: shape={data.shape}, exptime={exp_time:.2f}s, "
                  f"mean DN/s={np.nanmean(dn_array[:,:,i]):.2f}")
    
    return dn_array, edn_array, exp_times


def create_aia_temperature_response(wavelengths=['94', '131', '171', '193', '211', '335']):
    """
    Create temperature response functions for AIA channels
    
    This uses simplified/approximate response functions based on typical AIA characteristics.
    For production use, proper response functions from CHIANTI/SSW should be used.
    
    Returns:
        tresp: Temperature response matrix (n_temp x n_filters)
        tresp_logt: Log10 of temperatures
    """
    # Temperature grid (log10(T) from 5.5 to 7.5, covering ~300,000K to ~30MK)
    tresp_logt = np.arange(5.5, 7.51, 0.05)
    temperatures = 10**tresp_logt
    n_temp = len(tresp_logt)
    n_filters = len(wavelengths)
    
    # Peak response temperatures (log10 K) and widths for each AIA channel
    # These are approximate values based on AIA characteristics
    response_params = {
        '94': (6.85, 0.35),    # ~7 MK, hot flare plasma
        '131': (6.35, 0.45),   # ~2 MK and ~10 MK (dual peak, simplified to single)
        '171': (5.85, 0.25),   # ~0.7 MK, quiet corona
        '193': (6.15, 0.30),   # ~1.5 MK, active region
        '211': (6.30, 0.25),   # ~2 MK, active region
        '304': (4.90, 0.35),   # ~0.08 MK, chromosphere/transition region
        '335': (6.45, 0.30),   # ~2.5 MK, active region
    }
    
    tresp = np.zeros((n_temp, n_filters))
    
    print("\nCreating temperature response functions...")
    for i, wl in enumerate(wavelengths):
        if wl in response_params:
            peak_logt, width = response_params[wl]
            # Gaussian-like response
            tresp[:, i] = np.exp(-0.5 * ((tresp_logt - peak_logt) / width)**2)
            # Scale to reasonable values (~1e-26 to 1e-24 cm^5 DN/s)
            tresp[:, i] *= 1e-25
            print(f"  {wl}Å: peak at log(T)={peak_logt:.2f}, width={width:.2f}")
    
    return tresp, tresp_logt


def compute_dem_subregion(dn_array, edn_array, tresp, tresp_logt, 
                          region_slice=None, use_subset=True):
    """
    Compute DEM for a subregion of the image
    
    Args:
        dn_array: Full DN array
        edn_array: Full error array
        tresp: Temperature response matrix
        tresp_logt: Log temperatures for response
        region_slice: Tuple of slices for subregion (default: center 100x100)
        use_subset: If True, use smaller region for faster baseline
    
    Returns:
        dem, edem, elogt, chisq, dn_reg: DEM calculation outputs
        region_slice: The actual slice used
    """
    # Define subregion for faster computation
    if region_slice is None:
        if use_subset:
            # Use center 100x100 pixels for baseline
            center_y, center_x = dn_array.shape[0]//2, dn_array.shape[1]//2
            size = 500
            region_slice = (slice(center_y-size//2, center_y+size//2),
                           slice(center_x-size//2, center_x+size//2))
            print(f"\nUsing subregion: {size}x{size} pixels from center")
        else:
            # Use full image
            region_slice = (slice(None), slice(None))
            print("\nProcessing full image...")
    
    # Extract subregion
    dn_region = dn_array[region_slice]
    edn_region = edn_array[region_slice]
    
    print(f"DN region shape: {dn_region.shape}")
    
    # Temperature bins for DEM calculation
    temps = 10**np.arange(5.7, 7.0, 0.1)  # ~0.5 MK to 10 MK
    
    print(f"Temperature bins: {len(temps)} bins from {temps[0]/1e6:.2f} MK to {temps[-1]/1e6:.2f} MK")
    print("\nStarting DEM calculation...")
    
    start_time = time.time()
    
    # Call the DEM calculation
    dem, edem, elogt, chisq, dn_reg = dn2dem_pos(
        dn_region, 
        edn_region,
        tresp, 
        tresp_logt, 
        temps,
        reg_tweak=1.0,
        max_iter=10,
        rgt_fact=1.5,
        warn=False
    )
    
    elapsed_time = time.time() - start_time
    
    n_pixels = dn_region.shape[0] * dn_region.shape[1]
    print(f"\nDEM calculation complete!")
    print(f"  Elapsed time: {elapsed_time:.2f} seconds")
    print(f"  Pixels processed: {n_pixels}")
    print(f"  Time per pixel: {elapsed_time/n_pixels*1000:.2f} ms")
    print(f"  DEM shape: {dem.shape}")
    print(f"  Mean chi-squared: {np.nanmean(chisq):.3f}")
    
    return dem, edem, elogt, chisq, dn_reg, region_slice


def visualize_results(dem, aia_data, region_slice, output_dir):
    """
    Create visualizations of the DEM results
    
    Args:
        dem: DEM array (nx, ny, nt)
        aia_data: Original AIA data
        region_slice: Slice used for computation
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating visualizations...")
    
    # Calculate emission measure weighted median temperature
    temps = 10**np.arange(5.7, 7.0, 0.1)
    logt_centers = np.log10((temps[:-1] + temps[1:]) / 2)
    
    # For each pixel, calculate weighted median temperature
    em_weighted_temp = np.zeros(dem.shape[:2])
    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            dem_profile = dem[i, j, :]
            if np.any(dem_profile > 0):
                # Use DEM as weights for temperature
                weights = np.maximum(dem_profile, 0)
                if np.sum(weights) > 0:
                    em_weighted_temp[i, j] = np.average(logt_centers, weights=weights)
    
    # Create figure with DEM temperature map
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Original AIA 193 image (if available)
    if '193' in aia_data:
        im0 = axes[0].imshow(aia_data['193']['data'][region_slice], 
                             cmap='gray', origin='lower', 
                             vmin=0, vmax=np.percentile(aia_data['193']['data'][region_slice], 99))
        axes[0].set_title('AIA 193Å')
        axes[0].set_xlabel('X [pixels]')
        axes[0].set_ylabel('Y [pixels]')
        plt.colorbar(im0, ax=axes[0], label='DN/s')
    
    # Plot 2: DEM temperature map
    im1 = axes[1].imshow(em_weighted_temp, cmap='hot', origin='lower',
                         vmin=5.8, vmax=6.8)
    axes[1].set_title('DEM Emission Weighted Temperature')
    axes[1].set_xlabel('X [pixels]')
    axes[1].set_ylabel('Y [pixels]')
    cbar = plt.colorbar(im1, ax=axes[1], label='log(T) [K]')
    
    plt.tight_layout()
    output_file = output_dir / 'dem_baseline_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Create DEM profile plot for center pixel
    fig, ax = plt.subplots(figsize=(10, 6))
    center_i, center_j = dem.shape[0]//2, dem.shape[1]//2
    dem_profile = dem[center_i, center_j, :]
    
    ax.plot(logt_centers, dem_profile, 'b-', linewidth=2, label='DEM(T)')
    ax.set_xlabel('log(T) [K]', fontsize=12)
    ax.set_ylabel('DEM [cm⁻⁵ K⁻¹]', fontsize=12)
    ax.set_title(f'DEM Profile (pixel [{center_i}, {center_j}])', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    output_file = output_dir / 'dem_profile_center.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    return em_weighted_temp


def main():
    """
    Main pipeline for baseline DEMREG calculation
    """
    print("="*70)
    print("DEMREG BASELINE PIPELINE")
    print("="*70)
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    output_dir = project_dir / 'output'
    
    # Step 1: Load AIA data
    aia_data = load_aia_data(data_dir)
    
    # Step 2: Prepare DN arrays (use 6 channels)
    wavelengths = ['94', '131', '171', '193', '211', '335']
    dn_array, edn_array, exp_times = prepare_dn_data(aia_data, wavelengths)
    
    # Step 3: Create temperature response functions
    tresp, tresp_logt = create_aia_temperature_response(wavelengths)
    
    # Step 4: Compute DEM for subregion (100x100 for baseline)
    dem, edem, elogt, chisq, dn_reg, region_slice = compute_dem_subregion(
        dn_array, edn_array, tresp, tresp_logt, use_subset=True
    )
    
    # Step 5: Visualize results
    em_weighted_temp = visualize_results(dem, aia_data, region_slice, output_dir)
    
    # Save numerical results
    print("\nSaving numerical results...")
    np.savez(output_dir / 'dem_baseline_results.npz',
             dem=dem, edem=edem, elogt=elogt, chisq=chisq,
             em_weighted_temp=em_weighted_temp)
    print(f"  Saved: {output_dir / 'dem_baseline_results.npz'}")
    
    print("\n" + "="*70)
    print("BASELINE PIPELINE COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nNext steps for HPC acceleration:")
    print("  1. Profile the code to identify bottlenecks")
    print("  2. Implement CPU vectorization (NumPy/Numba)")
    print("  3. Implement GPU acceleration (CuPy/CUDA)")
    print("  4. Scale to full image processing")


if __name__ == '__main__':
    main()
