// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Ikhyeon Cho <tre0430@korea.ac.kr>

/*
 * moving_average_estimation.hpp
 *
 * Height estimation using exponential moving average (EMA).
 * Weights recent measurements more heavily, making it responsive to
 * changes in dynamic environments at the cost of statistical optimality.
 */

#ifndef FASTDEM_MAPPING_MOVING_AVERAGE_ESTIMATION_HPP
#define FASTDEM_MAPPING_MOVING_AVERAGE_ESTIMATION_HPP

#include <algorithm>
#include <cassert>
#include <cmath>

#include "fastdem/elevation_map.hpp"

namespace fastdem {

/**
 * @brief Elevation updater using exponential moving average (EMA).
 *
 * Update rule: elevation = (1 - alpha) * elevation + alpha * measurement
 *
 * Higher alpha gives more weight to new measurements (more responsive).
 * Lower alpha gives more weight to history (more stable).
 *
 * Layers created:
 * - elevation:    EMA height estimate
 * - n_points:     Measurement count
 * - upper_bound:  Set equal to elevation (no variance tracked)
 * - lower_bound:  Set equal to elevation (no variance tracked)
 *
 * Note: variance layer is not populated (EMA does not produce uncertainty).
 */
class MovingAverage {
 public:
  MovingAverage() = default;

  /**
   * @brief Construct with EMA weight.
   * @param alpha Weight for new measurements, clamped to [0.1, 0.9].
   */
  explicit MovingAverage(float alpha)
      : alpha_(std::clamp(alpha, 0.1f, 0.9f)) {}

  /// Create required layers on the map. Call once before first update.
  void ensureLayers(ElevationMap& map) {
    if (!map.exists(layer::n_points)) map.add(layer::n_points, 0.0f);
    if (!map.exists(layer::upper_bound)) map.add(layer::upper_bound, NAN);
    if (!map.exists(layer::lower_bound)) map.add(layer::lower_bound, NAN);
  }

  /// Cache matrix pointers. Call each frame before update loop.
  void bind(ElevationMap& map) {
    elevation_mat_ = &map.get(layer::elevation);
    count_mat_ = &map.get(layer::n_points);
    upper_mat_ = &map.get(layer::upper_bound);
    lower_mat_ = &map.get(layer::lower_bound);
    bound_ = true;
  }

  /// Update EMA estimate at a single cell.
  void update(const grid_map::Index& index, float measurement,
              [[maybe_unused]] float measurement_variance) {
    assert(bound_ && "MovingAverage::bind() must be called before update()");
    const int i = index(0);
    const int j = index(1);

    float& x = (*elevation_mat_)(i, j);
    float& count = (*count_mat_)(i, j);

    if (std::isnan(x)) {
      x = measurement;
      count = 1.0f;
    } else {
      count += 1.0f;
      x = (1.0f - alpha_) * x + alpha_ * measurement;
    }
  }

  /// Set bounds equal to elevation (no variance estimate available).
  void computeBounds(const grid_map::Index& index) {
    assert(bound_ &&
           "MovingAverage::bind() must be called before computeBounds()");
    const int i = index(0);
    const int j = index(1);
    (*upper_mat_)(i, j) = (*elevation_mat_)(i, j);
    (*lower_mat_)(i, j) = (*elevation_mat_)(i, j);
  }

 private:
  float alpha_ = 0.8f;

  grid_map::Matrix* elevation_mat_ = nullptr;
  grid_map::Matrix* count_mat_ = nullptr;
  grid_map::Matrix* upper_mat_ = nullptr;
  grid_map::Matrix* lower_mat_ = nullptr;

  bool bound_ = false;
};

}  // namespace fastdem

#endif  // FASTDEM_MAPPING_MOVING_AVERAGE_ESTIMATION_HPP
