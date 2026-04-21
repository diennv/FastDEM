// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Ikhyeon Cho <tre0430@korea.ac.kr>

/*
 * stat_mean_estimation.hpp
 *
 * Height estimation using Welford's online mean/variance algorithm.
 * Produces unbiased sample mean and variance updated incrementally.
 * Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 */

#ifndef FASTDEM_MAPPING_STAT_MEAN_ESTIMATION_HPP
#define FASTDEM_MAPPING_STAT_MEAN_ESTIMATION_HPP

#include <algorithm>
#include <cassert>
#include <cmath>

#include "fastdem/elevation_map.hpp"

namespace fastdem {

namespace layer {
constexpr auto stat_m2 = "_stat_m2";
}  // namespace layer

/**
 * @brief Elevation updater using Welford's online mean/variance algorithm.
 *
 * Incrementally computes the unbiased sample mean (elevation) and sample
 * variance using a numerically stable two-pass Welford accumulator.
 *
 * Layers created:
 * - elevation:    Sample mean
 * - variance:     Unbiased sample variance (M2 / (n-1))
 * - n_points:     Measurement count
 * - upper_bound:  elevation + 2*sigma
 * - lower_bound:  elevation - 2*sigma
 *
 * Internal layers:
 * - _stat_m2: Welford M2 accumulator (sum of squared deviations from mean)
 */
class StatMean {
 public:
  StatMean() = default;

  /// Create required layers on the map. Call once before first update.
  void ensureLayers(ElevationMap& map) {
    if (!map.exists(layer::variance)) map.add(layer::variance, 0.0f);
    if (!map.exists(layer::n_points)) map.add(layer::n_points, 0.0f);
    if (!map.exists(layer::stat_m2)) map.add(layer::stat_m2, 0.0f);
    if (!map.exists(layer::upper_bound)) map.add(layer::upper_bound, NAN);
    if (!map.exists(layer::lower_bound)) map.add(layer::lower_bound, NAN);
  }

  /// Cache matrix pointers. Call each frame before update loop.
  void bind(ElevationMap& map) {
    elevation_mat_ = &map.get(layer::elevation);
    variance_mat_ = &map.get(layer::variance);
    count_mat_ = &map.get(layer::n_points);
    m2_mat_ = &map.get(layer::stat_m2);
    upper_mat_ = &map.get(layer::upper_bound);
    lower_mat_ = &map.get(layer::lower_bound);
    bound_ = true;
  }

  /// Update mean/variance estimate at a single cell.
  void update(const grid_map::Index& index, float measurement,
              [[maybe_unused]] float measurement_variance) {
    assert(bound_ && "StatMean::bind() must be called before update()");
    const int i = index(0);
    const int j = index(1);

    float& x = (*elevation_mat_)(i, j);
    float& var = (*variance_mat_)(i, j);
    float& count = (*count_mat_)(i, j);
    float& m2 = (*m2_mat_)(i, j);

    if (std::isnan(x)) {
      x = measurement;
      count = 1.0f;
      m2 = 0.0f;
      var = 0.0f;
    } else {
      count += 1.0f;
      const float delta = measurement - x;
      x += delta / count;
      const float delta2 = measurement - x;
      m2 += delta * delta2;
      var = (count > 1.0f) ? m2 / (count - 1.0f) : 0.0f;
    }
  }

  /// Compute confidence bounds at a single cell.
  void computeBounds(const grid_map::Index& index) {
    assert(bound_ && "StatMean::bind() must be called before computeBounds()");
    const int i = index(0);
    const int j = index(1);
    const float sigma = std::sqrt(std::max(0.0f, (*variance_mat_)(i, j)));
    (*upper_mat_)(i, j) = (*elevation_mat_)(i, j) + 2.0f * sigma;
    (*lower_mat_)(i, j) = (*elevation_mat_)(i, j) - 2.0f * sigma;
  }

 private:
  grid_map::Matrix* elevation_mat_ = nullptr;
  grid_map::Matrix* variance_mat_ = nullptr;
  grid_map::Matrix* count_mat_ = nullptr;
  grid_map::Matrix* m2_mat_ = nullptr;
  grid_map::Matrix* upper_mat_ = nullptr;
  grid_map::Matrix* lower_mat_ = nullptr;

  bool bound_ = false;
};

}  // namespace fastdem

#endif  // FASTDEM_MAPPING_STAT_MEAN_ESTIMATION_HPP
