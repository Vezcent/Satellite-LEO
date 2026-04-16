/*
 * S-MAS: NRLMSISE-00 Simplified Atmospheric Density Model
 * Task 1.3 — Computes thermospheric density at LEO altitudes.
 *
 * This is a physically-motivated approximation of the full NRLMSISE-00
 * empirical model. It captures the key dependencies:
 *   - Exponential altitude profile with variable scale height
 *   - Solar activity (F10.7) → exospheric temperature → density
 *   - Geomagnetic activity (Ap) → thermospheric heating
 *   - Diurnal variation (day/night density asymmetry)
 *   - Latitudinal variation
 */
#pragma once

namespace smas {

class NRLMSISEModel {
public:
    // Compute atmospheric density (kg/m³) at given conditions.
    //   altitude_km :  height above Earth surface
    //   lat_deg     :  geodetic latitude (-90..+90)
    //   lst_hours   :  local solar time (0..24)
    //   f107        :  daily F10.7 index (sfu)
    //   f107a       :  81-day average F10.7 (use same if unavailable)
    //   ap          :  3-hour Ap index
    double density(double altitude_km,
                   double lat_deg,
                   double lst_hours,
                   double f107,
                   double f107a,
                   double ap) const;

private:
    // Exospheric temperature model
    double exospheric_temp(double f107, double f107a, double ap) const;

    // Temperature at altitude using Bates-Walker profile
    double temperature_at_altitude(double altitude_km, double T_inf) const;

    // Scale height at altitude
    double scale_height(double altitude_km, double T) const;
};

} // namespace smas
