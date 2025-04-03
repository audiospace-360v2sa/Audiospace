import os
import numpy as np
import librosa
from tqdm import tqdm

def load_foa_data(file_path):
    """
    Load FOA audio file and return scalar intensities I_x, I_y, I_z
    """
    data, sample_rate = librosa.load(file_path, sr=None, mono=False)
    if data.ndim == 1:
        raise ValueError(f"File {file_path} has insufficient channels, at least 4 channels (W, X, Y, Z) required")
    if data.ndim == 2:
        data = data.T  # Transpose to (samples, channels)
    if data.shape[1] < 4:
        raise ValueError(f"File {file_path} has insufficient channels, at least 4 channels (W, X, Y, Z) required")
    
    # Extract channels
    W, X, Y, Z = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    
    # Calculate scalar intensity vectors
    I_x = np.mean(W * X)  # Mean of dot product between W and X
    I_y = np.mean(W * Y)  # Mean of dot product between W and Y
    I_z = np.mean(W * Z)  # Mean of dot product between W and Z
    
    return I_x, I_y, I_z

def calculate_theta_phi(I_x, I_y, I_z):
    """
    Calculate θ (azimuth), φ (elevation) based on scalar intensities I_x, I_y, I_z
    """
    theta = np.arctan2(I_y, I_x)
    phi = np.arctan2(I_z, np.sqrt(np.maximum(I_x**2 + I_y**2, 1e-6)))
    return theta, phi

def circular_l1_error(theta_gt, theta_gen):
    """Calculate L1 error for circular angles"""
    diff = np.abs(theta_gt - theta_gen)
    return np.minimum(diff, 2 * np.pi - diff)

def calculate_errors(ground_truth_dir1, generated_dir, matched_files=None, error_type="MAE"):
    """Calculate errors for all matched files"""

    theta_errors = []
    phi_errors = []
    spatial_angle_errors = []

    for file_name in tqdm(matched_files):
        gt_file_path = os.path.join(ground_truth_dir1, file_name[0])
        gen_file_path = os.path.join(generated_dir, file_name[1])

        # Load and calculate scalar intensity vectors
        I_x_gt, I_y_gt, I_z_gt = load_foa_data(gt_file_path)
        I_x_gen, I_y_gen, I_z_gen = load_foa_data(gen_file_path)

        # Calculate θ, φ
        theta_gt, phi_gt = calculate_theta_phi(I_x_gt, I_y_gt, I_z_gt)
        theta_gen, phi_gen = calculate_theta_phi(I_x_gen, I_y_gen, I_z_gen)

        # Calculate errors
        if error_type =='MAE':
            theta_errors.append(np.abs(circular_l1_error(theta_gt, theta_gen)))
            phi_errors.append(np.abs(phi_gt - phi_gen))
        elif error_type == 'MSE':
            theta_errors.append(circular_l1_error(theta_gt, theta_gen) ** 2)
            phi_errors.append((phi_gt - phi_gen) ** 2)
        else:
            raise ValueError("error_type must be one of ['MAE', 'MSE']")

        # Calculate Spatial Angle
        delta_phi = phi_gt - phi_gen
        delta_theta = theta_gt - theta_gen
        a = (np.sin(delta_phi / 2)**2 + 
             np.cos(phi_gt) * np.cos(phi_gen) * np.sin(delta_theta / 2)**2)
        a = np.clip(a, 0, 1)
        spatial_angle = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        if error_type =='MAE':
            spatial_angle_errors.append(np.abs(spatial_angle))
        else: 
            spatial_angle_errors.append(spatial_angle ** 2)

    return (
            np.mean(theta_errors),
            np.mean(phi_errors),
            np.mean(spatial_angle_errors)
        )

def calculate_spatial_metrics(reference_path, generated_path, split_path, error_type="MAE"):
    """
    Calculate spatial metrics between reference and generated audio files

    Args:
        reference_path: Path to reference audio files
        generated_path: Path to generated audio files
        split_path: Path to file containing list of files to process
        error_type: Type of error metric to use ( 'MAE', 'MSE')
    """
    if error_type not in ['MAE', 'MSE']:
        raise ValueError("error_type must be one of ['MAE', 'MSE']")

    if split_path:
        with open(split_path, "r") as f:
            base_files = set(f.read().splitlines())
            # Remove file extensions
            base_files = {os.path.splitext(f)[0] for f in base_files}
    else:
        # Get base names (without extensions) of all files in both directories
        ref_files = {os.path.splitext(f)[0] for f in os.listdir(reference_path)}
        gen_files = {os.path.splitext(f)[0] for f in os.listdir(generated_path)}
        base_files = ref_files & gen_files

    matched_files = []
    for base in base_files:
        # Search for reference file with supported extensions
        ref_path = None
        for ext in ['.flac', '.wav']:
            if os.path.exists(os.path.join(reference_path, base + ext)):
                ref_path = base + ext
                break
               
        # Search for generated file with supported extensions
        gen_path = None
        for ext in ['.flac', '.wav']:
            if os.path.exists(os.path.join(generated_path, base + ext)):
                gen_path = base + ext
                break
               
        if ref_path and gen_path:
            matched_files.append((ref_path, gen_path))

    avg_theta, avg_phi, avg_spatial = calculate_errors(
        reference_path, generated_path, matched_files, error_type)

    return avg_theta, avg_phi, avg_spatial