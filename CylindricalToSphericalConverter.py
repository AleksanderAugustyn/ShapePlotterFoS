"""
Cylindrical to Spherical Coordinate Converter for Nuclear Shapes
Converts nuclear shapes from cylindrical coordinates ρ(z) to spherical coordinates r(θ)
"""

from typing import Tuple, List, Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, brentq


class CylindricalToSphericalConverter:
    """
    Converts nuclear shapes from cylindrical coordinates ρ(z) to spherical coordinates r(θ).

    Handles non-convex shapes by using parametric boundary tracing.

    In cylindrical coordinates:
    - z: axial coordinate
    - ρ: radial coordinate perpendicular to z-axis

    In spherical coordinates:
    - r: radial distance from origin
    - θ: polar angle from positive z-axis (0 to π)
    - φ: azimuthal angle (not used due to axial symmetry)

    Relationships:
    - z = r cos(θ)
    - ρ = r sin(θ)
    - r = √(z² + ρ²)
    """

    def __init__(self, z_points: np.ndarray, rho_points: np.ndarray):
        """
        Initialize the converter with shape data in cylindrical coordinates.

        Args:
            z_points: Array of z coordinates
            rho_points: Array of corresponding ρ values
        """
        # Store the original data
        self.z_points = np.asarray(z_points)
        self.rho_points = np.asarray(rho_points)

        # Ensure points are sorted by z
        sort_idx = np.argsort(self.z_points)
        self.z_points = self.z_points[sort_idx]
        self.rho_points = self.rho_points[sort_idx]

        # Create an interpolation function for ρ(z)
        self.rho_interp = interp1d(
            self.z_points,
            self.rho_points,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )

        # Store shape boundaries
        self.z_min = np.min(self.z_points[self.rho_points > 0])
        self.z_max = np.max(self.z_points[self.rho_points > 0])

    def rho_of_z(self, z: float) -> float:
        """
        Get the radial coordinate ρ for a given axial coordinate z.

        Args:
            z: Axial coordinate

        Returns:
            ρ: Radial coordinate at z
        """
        if z < self.z_min or z > self.z_max:
            return 0.0
        return float(self.rho_interp(z))

    def _solve_r_for_theta_all_intersections(self, theta: float, n_samples: int = 1000) -> List[float]:
        """
        Find ALL r values for a given θ by sampling along rays.

        This method handles non-convex shapes by finding all intersections
        of a ray at an angle θ with the shape boundary.

        Args:
            theta: Polar angle in radians (0 to π)
            n_samples: Number of samples along the ray

        Returns:
            List of r values where the ray intersects the shape
        """
        # Handle special cases
        if abs(theta) < 1e-10:  # North Pole
            return [abs(self.z_max)]
        elif abs(theta - np.pi) < 1e-10:  # South Pole
            return [abs(self.z_min)]

        # For general θ, sample along the ray
        r_max: float = 2 * max(abs(self.z_max), abs(self.z_min), np.max(self.rho_points))
        r_samples = np.linspace(0, r_max, n_samples)

        intersections = []
        prev_inside = False
        prev_r = 0.0

        for i, r in enumerate(r_samples):
            z: float = r * np.cos(theta)
            rho_ray: float = r * np.sin(theta)

            # Check if this point is inside the shape
            if self.z_min <= z <= self.z_max:
                rho_shape: float = self.rho_of_z(z)
                inside: bool = rho_ray <= rho_shape

                # Detect crossing
                if inside and not prev_inside and r > 0:
                    # Entering the shape - refine the intersection
                    r_refined: float = self._refine_intersection(theta, prev_r, r)
                    if r_refined > 0:
                        intersections.append(r_refined)
                elif not inside and prev_inside:
                    # Exiting the shape - refine the intersection
                    r_refined = self._refine_intersection(theta, prev_r, r)
                    if r_refined > 0:
                        intersections.append(r_refined)

                prev_inside = inside
            else:
                # Outside the valid z range
                if prev_inside:
                    # Was inside, now outside - add exit intersection
                    r_refined = self._refine_intersection(theta, prev_r, r)
                    if r_refined > 0:
                        intersections.append(r_refined)
                prev_inside = False

            prev_r = r

        # Remove duplicates and sort
        if intersections:
            intersections = sorted(list(set(np.round(intersections, decimals=6))))

        return intersections

    def _refine_intersection(self, theta: float, r_low: float, r_high: float) -> float:
        """
        Refine the intersection point between r_low and r_high.

        Args:
            theta: Polar angle
            r_low: Lower bound for r
            r_high: Upper bound for r

        Returns:
            Refined r value at intersection
        """

        def equation(r):
            z = r * np.cos(theta)
            if z < self.z_min or z > self.z_max:
                return r  # Outside shape
            return r * np.sin(theta) - self.rho_of_z(z)

        try:
            # Use Brent's method if we have a sign change
            f_low = equation(r_low)
            f_high = equation(r_high)

            if f_low * f_high < 0:
                return brentq(equation, r_low, r_high)
            else:
                # No sign change, use the closer value
                return r_low if abs(f_low) < abs(f_high) else r_high
        except:
            return (r_low + r_high) / 2

    def convert_to_spherical_parametric(self, n_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the shape to spherical coordinates using parametric tracing.

        This method traces along the shape boundary in cylindrical coordinates
        and converts each point to spherical coordinates. This naturally handles
        non-convex shapes and multiple-valued regions.

        Args:
            n_points: Number of points (if None, uses original resolution)

        Returns:
            theta_points: Array of polar angles
            r_points: Array of radial distances r(θ)
        """
        if n_points is None:
            z_trace = self.z_points
            rho_trace = self.rho_points
        else:
            # Resample the shape with uniform spacing
            z_trace = np.linspace(self.z_min, self.z_max, n_points)
            rho_trace = np.array([self.rho_of_z(z) for z in z_trace])

        # Remove points where rho = 0
        valid_mask = rho_trace > 1e-10
        z_trace = z_trace[valid_mask]
        rho_trace = rho_trace[valid_mask]

        # Convert to spherical coordinates
        r_points = np.sqrt(z_trace ** 2 + rho_trace ** 2)
        theta_points = np.arctan2(rho_trace, z_trace)

        # Ensure theta is in [0, π]
        theta_points = np.where(theta_points < 0, theta_points + np.pi, theta_points)

        # Sort by theta for consistency
        sort_idx = np.argsort(theta_points)
        theta_points = theta_points[sort_idx]
        r_points = r_points[sort_idx]

        return theta_points, r_points

    def convert_to_spherical_uniform_theta(self, n_theta: int = 180) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to spherical coordinates with uniform theta spacing.

        For non-convex shapes, this returns the OUTERMOST r value for each theta.
        This is typically what's needed for shape visualization.

        Args:
            n_theta: Number of θ points (from 0 to π)

        Returns:
            theta_points: Array of polar angles
            r_points: Array of radial distances r(θ)
        """
        theta_points = np.linspace(0, np.pi, n_theta)
        r_points = np.zeros(n_theta)

        for i, theta in enumerate(theta_points):
            # Get all intersections for this angle
            intersections = self._solve_r_for_theta_all_intersections(theta)

            if intersections:
                # Use the outermost intersection
                r_points[i] = max(intersections)
            else:
                # No intersection found - use old method as fallback
                r_points[i] = self._solve_r_for_theta(theta)

        return theta_points, r_points

    def _solve_r_for_theta(self, theta: float, initial_guess: Optional[float] = None) -> float:
        """
        Legacy method: Solve for r given θ using the implicit equation.
        Kept for backward compatibility and as fallback.

        Solve for r given θ using the implicit equation:
        r sin(θ) = ρ(r cos(θ))

        Args:
            theta: Polar angle in radians (0 to π)
            initial_guess: Initial guess for r

        Returns:
            r: Radial distance at an angle θ
        """
        # Handle special cases
        if theta == 0:  # North Pole
            return abs(self.z_max)
        elif theta == np.pi:  # South Pole
            return abs(self.z_min)
        elif abs(theta - np.pi / 2) < 1e-10:  # Equator
            return self.rho_of_z(0.0)

        # For general θ, we need to solve: r sin(θ) = ρ(r cos(θ))
        def equation(r):
            z = r * np.cos(theta)
            # Check if z is within the valid range
            if z < self.z_min or z > self.z_max:
                return r  # This makes r = 0 a solution outside the shape
            return r * np.sin(theta) - self.rho_of_z(z)

        # Get a good initial guess
        if initial_guess is None:
            # Try to estimate based on the maximum extent
            z_at_theta = self.z_max * np.cos(theta) if theta < np.pi / 2 else self.z_min * np.cos(theta)
            if self.z_min <= z_at_theta <= self.z_max:
                rho_at_z = self.rho_of_z(z_at_theta)
                if np.sin(theta) > 1e-10:
                    initial_guess = rho_at_z / np.sin(theta)
                else:
                    initial_guess = abs(z_at_theta / np.cos(theta))
            else:
                initial_guess = max(abs(self.z_max), abs(self.z_min))

        # Try to bracket the root
        try:
            # Find bounds for the root
            r_min = 0
            r_max = 2 * max(abs(self.z_max), abs(self.z_min), np.max(self.rho_points))

            # Check if there's a sign change
            f_min = equation(r_min)
            f_max = equation(r_max)

            if f_min * f_max < 0:
                # Use Brent's method if we have a bracketed root
                r_solution = brentq(equation, r_min, r_max)
            else:
                # Fall back to solve
                result = fsolve(equation, initial_guess, full_output=True)
                r_solution = result[0][0]

                # Check if a solution is valid
                if result[2] != 1 or r_solution < 0:
                    r_solution = 0.0
        except (RuntimeError, ValueError):
            # If numerical methods fail (e.g., non-convergence), default to 0
            r_solution = 0.0

        return max(0.0, r_solution)  # Ensure non-negative

    def convert_to_spherical(self, n_theta: int = 180, method: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the shape to spherical coordinates.

        Args:
            n_theta: Number of θ points (from 0 to π)
            method: Conversion method ('parametric' or 'uniform')

        Returns:
            theta_points: Array of polar angles
            r_points: Array of radial distances r(θ)
        """
        if method == 'parametric':
            return self.convert_to_spherical_parametric(n_theta)
        else:
            return self.convert_to_spherical_uniform_theta(n_theta)

    def convert_to_cartesian(self, n_theta: int = 180, method: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the shape to Cartesian coordinates (x, y) for 2D plotting.
        This gives the cross-section of the axially symmetric shape.

        Args:
            n_theta: Number of θ points
            method: Conversion method ('parametric' or 'uniform')

        Returns:
            x_points: Array of x coordinates (r sin(θ))
            y_points: Array of y coordinates (r cos(θ))
        """
        theta_points, r_points = self.convert_to_spherical(n_theta, method)

        # Convert to Cartesian for 2D cross-section
        x_points = r_points * np.sin(theta_points)
        y_points = r_points * np.cos(theta_points)

        return x_points, y_points

    def detect_multi_valued_regions(self, n_theta: int = 360) -> List[Tuple[float, float]]:
        """
        Detect angular regions where the shape is multivalued.

        Returns:
            List of (theta_start, theta_end) tuples indicating multivalued regions
        """
        theta_points = np.linspace(0, np.pi, n_theta)
        multi_valued_regions = []

        region_start = None
        for theta in theta_points:
            intersections = self._solve_r_for_theta_all_intersections(theta)

            if len(intersections) > 1:
                if region_start is None:
                    region_start = theta
            else:
                if region_start is not None:
                    multi_valued_regions.append((region_start, theta))
                    region_start = None

        # Handle case where a multivalued region extends to π
        if region_start is not None:
            multi_valued_regions.append((region_start, np.pi))

        return multi_valued_regions

    def get_max_radius(self) -> float:
        """
        Get the maximum radial distance of the shape.

        Returns:
            Maximum radius
        """
        # Sample many angles to find maximum
        theta_sample = np.linspace(0, np.pi, 361)
        r_sample = []

        for theta in theta_sample:
            intersections = self._solve_r_for_theta_all_intersections(theta)
            if intersections:
                r_sample.append(max(intersections))

        return np.max(r_sample) if r_sample else 0.0

    def get_shape_at_angle(self, theta: float) -> Tuple[float, float, float]:
        """
        Get the shape parameters at a specific angle θ.

        Args:
            theta: Polar angle in radians

        Returns:
            r: Radial distance
            z: Axial coordinate
            rho: Radial coordinate in a cylindrical system
        """
        intersections = self._solve_r_for_theta_all_intersections(theta)

        if intersections:
            r = max(intersections)  # Use outermost
        else:
            r = self._solve_r_for_theta(theta)

        z = r * np.cos(theta)
        rho = r * np.sin(theta)
        return r, z, rho

    def validate_conversion(self, n_samples: int = 200, method: str = 'uniform') -> dict:
        """
        Validate the conversion by checking consistency.

        Args:
            n_samples: Number of sample points to check
            method: Conversion method to validate

        Returns:
            Dictionary with validation metrics
        """
        theta_sample = np.linspace(0.07, np.pi - 0.07, n_samples)
        errors = []

        for theta in theta_sample:
            r, z, rho = self.get_shape_at_angle(theta)
            if r > 0:
                # Check if the calculated ρ matches the interpolated ρ(z)
                rho_expected = self.rho_of_z(z)
                if rho_expected > 0:
                    error = np.square(rho - rho_expected)
                    errors.append(error)

        # Also check for multivalued regions
        multi_valued_regions = self.detect_multi_valued_regions()

        return {
            'root_mean_squared_error': np.sqrt(np.mean(errors)) if errors else 0,
            'max_error': np.max(errors) if errors else 0,
            'n_valid_points': len(errors),
            'n_failed_points': n_samples - len(errors),
            'multi_valued_regions': multi_valued_regions,
            'has_multi_valued_regions': len(multi_valued_regions) > 0
        }
