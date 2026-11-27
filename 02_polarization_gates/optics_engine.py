"""
OptiCNN: Optical Field Simulation Engine (optics_engine.py)
===========================================================
Models 4 physical properties of light:
  1. Wavelength / Frequency -> Wavelength Division Multiplexing (WDM)
  2. Amplitude / Intensity   -> Carries energy and information (I = |E|^2)
  3. Phase / Interference    -> Coherent interference in space
  4. Polarization            -> Electric field vector orientation [Ex, Ey]^T
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

class OpticalField:
    """
    A single monochromatic optical field mode.
    """
    def __init__(self, wavelength_nm=1550.0, amplitude=1.0, phase_rad=0.0, jones_vector=None):
        self.wavelength_nm = float(wavelength_nm)  # Wavelength (default telecom C-band 1550 nm)
        self.amplitude = float(amplitude)          # Field amplitude (real >= 0)
        self.phase_rad = float(phase_rad % (2 * np.pi)) # Phase (0 to 2*pi)

        # Polarization state: normalized complex Jones vector [Ex, Ey]^T
        # [1, 0] is Horizontal (H), [0, 1] is Vertical (V)
        # [1, 1]/sqrt(2) is Diagonal (45 deg)
        if jones_vector is None:
            self.jones = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        else:
            j = np.array(jones_vector, dtype=np.complex128)
            norm = np.linalg.norm(j)
            self.jones = j / norm if norm > 1e-12 else np.array([1.0+0j, 0+0j])

    @property
    def intensity(self):
        """Optical intensity (power) I = |A|^2"""
        return self.amplitude ** 2

    @property
    def complex_field(self):
        """Full spatial complex electric field vector E = A * exp(i * phase) * Jones"""
        return self.amplitude * np.exp(1.0j * self.phase_rad) * self.jones

    def __repr__(self):
        h_power = np.abs(self.jones[0])**2
        v_power = np.abs(self.jones[1])**2
        return (f"OpticalField(lambda={self.wavelength_nm:.1f}nm, I={self.intensity:.3f}, "
                f"Phase={np.degrees(self.phase_rad):.1f} deg, H/V=[{h_power:.2f}, {v_power:.2f}])")


class LaserEmitter:
    """
    Laser source with tunable wavelength, intensity, phase, and polarization.
    """
    def __init__(self, wavelength_nm=1550.0):
        self.wavelength_nm = wavelength_nm

    def emit(self, intensity=1.0, phase_deg=0.0, polarization_angle_deg=0.0):
        """
        Emits a light beam:
        polarization_angle_deg: 0 deg = H, 90 deg = V, 45 deg = diagonal
        """
        theta = np.radians(polarization_angle_deg)
        jones = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
        return OpticalField(
            wavelength_nm=self.wavelength_nm,
            amplitude=np.sqrt(max(intensity, 0.0)),
            phase_rad=np.radians(phase_deg),
            jones_vector=jones
        )


class OpticalReceiver:
    """
    Optical receiver with multiple detection modes:
    1. Square-law intensity detection
    2. Polarization beam splitting (PBS: decomposes into H and V components)
    3. Spectrometer channel decomposition (WDM demultiplexing)
    """
    def detect_intensity(self, field: OpticalField):
        """Detects scalar optical intensity I = |E|^2"""
        return field.intensity

    def detect_polarization_split(self, field: OpticalField):
        """
        Detects H and V polarization intensities via Polarization Beam Splitter (PBS).
        Satisfies: I_H + I_V = I_total
        """
        total_I = field.intensity
        I_h = total_I * (np.abs(field.jones[0]) ** 2)
        I_v = total_I * (np.abs(field.jones[1]) ** 2)
        return {"I_H": I_h, "I_V": I_v, "Total": total_I}

    def combine_and_detect(self, fields: list[OpticalField]):
        """
        Combines multiple input optical beams:
        - Same wavelength + same polarization -> Coherent interference (amplitudes add)
        - Different wavelength OR orthogonal polarization -> Incoherent sum (intensities add)
        """
        wl_groups = {}
        for f in fields:
            wl = round(f.wavelength_nm, 2)
            wl_groups.setdefault(wl, []).append(f)

        total_measured_power = 0.0
        details = {}

        for wl, grp in wl_groups.items():
            # For same wavelength, sum complex field vectors coherently
            total_E = np.zeros(2, dtype=np.complex128)
            for f in grp:
                total_E += f.complex_field

            p_wl = np.sum(np.abs(total_E) ** 2)
            details[f"{wl}nm"] = p_wl
            total_measured_power += p_wl

        return {"total_power": total_measured_power, "channels": details}


class PolarizationGate:
    """
    Polarization gate:
    Modulates polarization state via electro-optic effect (Pockels Cell) or waveplates.
    """
    @staticmethod
    def rotate(field: OpticalField, theta_rad: float):
        """Rotates polarization angle (waveplate / polarization rotator)."""
        rot_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)]
        ], dtype=np.complex128)
        new_jones = rot_matrix @ field.jones
        return OpticalField(
            wavelength_nm=field.wavelength_nm,
            amplitude=field.amplitude,
            phase_rad=field.phase_rad,
            jones_vector=new_jones
        )

    @staticmethod
    def phase_retard(field: OpticalField, delta_phi_rad: float):
        """Introduces phase delay to vertical component relative to horizontal (birefringent crystal)."""
        retard_matrix = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, np.exp(1.0j * delta_phi_rad)]
        ], dtype=np.complex128)
        new_jones = retard_matrix @ field.jones
        return OpticalField(
            wavelength_nm=field.wavelength_nm,
            amplitude=field.amplitude,
            phase_rad=field.phase_rad,
            jones_vector=new_jones
        )
