"""
Beta Deformation Parameter Calculator
Calculates nuclear deformation parameters β_λ using spherical harmonic decomposition
"""

from typing import Dict, Tuple, Optional

import numpy as np
from scipy.integrate import simpson
from scipy.special import sph_harm_y


class BetaDeformationCalculator:
    """
    Calculate beta deformation parameters for nuclear shapes using spherical harmonics.

    The nuclear surface is expanded as:
    R(θ, φ) = c R₀ [1 + Σ_λμ β_λμ Y_λμ(θ, φ)]

    For axially symmetric shapes (m=0):
    R(θ) = c R₀ [1 + Σ_λ β_λ0 Y_λ0(θ)]
    where c is a volume fixing factor and R₀ is the radius of a sphere with the same volume as the nucleus.
    """

    def __init__(self, theta: np.ndarray, radius: np.ndarray, number_of_nucleons: int):
        """
        Initialize the calculator with shape data.

        Args:
            theta: Array of polar angles (0 to π)
            radius: Array of radial distances r(θ)
        """
        self.theta = np.asarray(theta)
        self.r = np.asarray(radius)
        self.r0 = 1.16
        self.radius0 = 1.16 * number_of_nucleons ** (1 / 3)

        # Ensure theta is sorted
        sort_idx = np.argsort(self.theta)
        self.theta = self.theta[sort_idx]
        self.r = self.r[sort_idx]

        # Precompute sin(theta) for integration
        self.sin_theta = np.sin(self.theta)

        # Cache for spherical harmonics
        self._ylm_cache = {}

    def _get_ylm(self, l: int, m: int = 0) -> np.ndarray:
        """
        Get spherical harmonic Y_lm(θ, φ) for m=0 (axial symmetry).

        For m=0, Y_l0 only depends on θ and is real.

        Args:
            l: Degree of spherical harmonic
            m: Order (default 0 for axial symmetry)

        Returns:
            Array of Y_l0(θ) values
        """
        key = (l, m)
        if key not in self._ylm_cache:
            # sph_harm_y uses (l, m, theta, phi) convention
            # For m=0, the result is independent of phi
            phi = 0.0
            self._ylm_cache[key] = sph_harm_y(l, m, self.theta, phi).real
        return self._ylm_cache[key]

    def calculate_beta_parameters(self, l_max: int = 12) -> Dict[int, float]:
        """
        Calculate β_λ0 deformation parameters.

        β_λ0 = sqrt(4π) * ∫ r(θ) Y_λ0(θ) sin(θ) dθ / ∫ r(θ) Y_00(θ) sin(θ) dθ

        Args:
            l_max: Maximum l value to calculate

        Returns:
            Dictionary of β_λ parameters
        """
        beta = {}

        # Calculate denominator: ∫ Y_00(θ) sin(θ) dθ
        # For l=0, Y_00 = 1/sqrt(4π)
        integrand_denominator = self.r * self._get_ylm(0, 0) * self.sin_theta
        denominator = simpson(integrand_denominator, x=self.theta)

        for l in range(1, l_max + 1):
            # Get spherical harmonic
            ylm = self._get_ylm(l, 0)

            # Calculate numerator: ∫ r(θ) Y_λ0(θ) sin(θ) dθ
            integrand_numerator = self.r * ylm * self.sin_theta
            numerator = simpson(integrand_numerator, x=self.theta)

            if abs(denominator) > 1e-10:
                beta[l] = np.sqrt(4 * np.pi) * numerator / denominator
            else:
                beta[l] = 0.0

        return beta

    @staticmethod
    def calculate_rms_deformation(beta: Dict[int, float], l_max: Optional[int] = None) -> float:
        """
        Calculate root-mean-square deformation.

        Δ_rms = sqrt(Σ_λ β_λ²)

        Args:
            beta: Dictionary of beta parameters
            l_max: Maximum l to include (None = all available)

        Returns:
            RMS deformation
        """
        if l_max is None:
            l_values = [l for l in beta.keys() if l > 0]
        else:
            l_values = [l for l in beta.keys() if 0 < l <= l_max]

        sum_squared = sum(beta[l] ** 2 for l in l_values)
        return np.sqrt(sum_squared)

    @staticmethod
    def calculate_volume_in_spherical_coordinates(radius: np.ndarray, theta: np.ndarray) -> float:
        """
        Calculate the volume of the shape in spherical coordinates.

        V = ∫ r(θ)² sin(θ) dr dθ dφ = ∫ r(θ)² sin(θ) dr dθ * 2π

        Args:
            radius: Array of radial distances r(θ)
            theta: Array of polar angles θ

        Returns:
            Volume of the shape
        """
        # Volume integral in spherical coordinates
        integrand = radius ** 3 * np.sin(theta)
        volume = simpson(integrand, x=theta) * 2 / 3 * np.pi

        return volume

    @staticmethod
    def calculate_center_of_mass_in_spherical_coordinates(radius: np.ndarray, theta: np.ndarray) -> float:
        """
        Calculate the center of mass in spherical coordinates.

        CM = ∫ r(θ)² sin(θ) dr dθ dφ / ∫ r(θ) sin(θ) dr dθ dφ

        Args:
            radius: Array of radial distances r(θ)
            theta: Array of polar angles θ

        Returns:
            Center of mass in spherical coordinates
        """
        # Numerator: 2π ∫ r(θ)³ sin(θ) cos(θ) dr dθ = 1/2 π r(θ)⁴ ∫ sin(θ) cos(θ) dθ
        numerator = simpson(1 / 2 * np.pi * radius ** 4 * np.sin(theta) * np.cos(theta), x=theta)

        # Denominator: Volume integral = 2π ∫ r(θ)² sin(θ) dr dθ
        denominator = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(radius, theta)

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def reconstruct_shape(self, beta: Dict[int, float], n_theta: int = 180) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the shape from beta parameters.

        R(θ) = c R₀ [1 + Σ_λ β_λ Y_λ0(θ)]

        Args:
            beta: Dictionary of beta parameters
            n_theta: Number of theta points for reconstruction

        Returns:
            theta_reconstructed: Array of theta values
            radius_reconstructed: Array of reconstructed radii
        """
        theta_reconstructed = np.linspace(0, np.pi, n_theta)
        radius_pre_normalization = np.ones_like(theta_reconstructed) * self.radius0

        for l, beta_l in beta.items():
            if l == 0:
                continue  # Skip l=0 for shape reconstruction

            # Get Y_l0 at reconstruction points
            phi = 0
            ylm = sph_harm_y(l, 0, theta_reconstructed, phi).real

            # Add contribution
            radius_pre_normalization += self.radius0 * beta[l] * ylm

        # Calculate volume fixing factor
        volume_pre_normalization = self.calculate_volume_in_spherical_coordinates(radius_pre_normalization, theta_reconstructed)
        sphere_volume = (4 / 3) * np.pi * self.radius0 ** 3
        volume_fixing_factor = sphere_volume / volume_pre_normalization

        # Calculate radius fixing factor
        radius_fixing_factor = volume_fixing_factor ** (1 / 3)

        radius_reconstructed = radius_fixing_factor * radius_pre_normalization

        return theta_reconstructed, radius_reconstructed
