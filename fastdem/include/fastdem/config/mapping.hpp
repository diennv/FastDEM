// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Ikhyeon Cho <tre0430@korea.ac.kr>

#ifndef FASTDEM_CONFIG_MAPPING_HPP
#define FASTDEM_CONFIG_MAPPING_HPP

namespace fastdem {

/// Map origin behavior.
enum class MappingMode {
  LOCAL,  ///< Follows robot (robot-centric)
  GLOBAL  ///< Fixed origin (world frame)
};

/// Height estimation algorithm.
enum class EstimationType {
  Kalman,        ///< Recursive Bayesian (Kalman filter)
  P2Quantile,    ///< Online quantile (P² algorithm)
  StatMean,      ///< Welford online mean/variance
  MovingAverage, ///< Exponential moving average
};

namespace config {

/// Kalman filter parameters. Measurement variance comes from sensor_model.
struct Kalman {
  float min_variance = 0.0001f;  ///< Floor (prevents over-confidence)
  float max_variance = 0.01f;    ///< Cap (bounds uncertainty)
  float process_noise = 0.0f;    ///< Q (0 = static environment)
};

/// P² quantile estimator parameters.
/// Default markers: {1%, 16%, 50%, 84%, 99%}.
struct P2Quantile {
  float dn0 = 0.01f;              ///< 1st percentile (soft min)
  float dn1 = 0.16f;              ///< 16th percentile
  float dn2 = 0.50f;              ///< 50th percentile (median)
  float dn3 = 0.84f;              ///< 84th percentile
  float dn4 = 0.99f;              ///< 99th percentile (soft max)
  int elevation_marker = 3;       ///< Which marker for elevation (0-4)
  float max_sample_count = 0.0f;  ///< Fading memory limit (0 = disabled)
};

/// MovingAverage estimator parameters.
struct MovingAverage {
  float alpha = 0.8f;  ///< EMA weight [0.1, 0.9]: higher = more weight to new measurement
};

/// Mapping configuration (consumed by ElevationMapping).
struct Mapping {
  MappingMode mode = MappingMode::LOCAL;
  EstimationType estimation_type = EstimationType::Kalman;
  Kalman kalman;
  P2Quantile p2;
  MovingAverage moving_average;
};

}  // namespace config
}  // namespace fastdem

#endif  // FASTDEM_CONFIG_MAPPING_HPP
