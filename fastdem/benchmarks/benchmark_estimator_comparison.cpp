// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Ikhyeon Cho <tre0430@korea.ac.kr>

// Estimator Comparison Benchmark
//
// Compares four height estimators across three dimensions:
//   A. Timing Performance     - wall-clock cost per scan (vary obs count & resolution)
//   B. Convergence Accuracy   - RMSE vs scan count under additive noise
//   C. Dynamic Response       - recovery latency after a step height change
//   D. Memory Footprint       - extra layers and bytes per cell per estimator
//
// Estimators:
//   - Kalman        : 1D Kalman filter with Welford sample-variance side channel
//   - P2Quantile    : Online quantile (P² algorithm), elevation = 84th percentile
//   - StatMean      : Welford online mean/variance (unbiased, parameter-free)
//   - MovingAverage : Exponential moving average (responsive, no variance)
//
// Outputs (stdout):
//   ASCII tables for all four sections.
//
// CSV outputs (current directory):
//   convergence_rmse.csv   - per-scan RMSE for Section B (plot convergence curves)
//   dynamic_response.csv   - per-scan RMSE for Section C (plot step response)
//
// Build:
//   cmake .. -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
//   make -j benchmark_estimator_comparison
//
// Run:
//   ./benchmark_estimator_comparison
//
// Expected runtime: ~1-3 minutes (depends on hardware)
//
// Notes:
//   - All synthetic data; no external dataset required.
//   - RNG seeded to 42 for reproducibility.
//   - P2Quantile requires >=5 measurements per cell before outputting;
//     RMSE for scans 1-4 is shown as "-" in the convergence table.
//   - P2Quantile (84th pct) has an upward bias of ~+sigma on zero-mean noise;
//     this is expected behavior and documented in the output.
//   - MovingAverage does not track variance; upper/lower bounds equal elevation.

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <grid_map_core/grid_map_core.hpp>

#include "../lib/nanoPCL/benchmarks/common/benchmark_common.hpp"
#include "fastdem/elevation_map.hpp"
#include "fastdem/mapping/kalman_estimation.hpp"
#include "fastdem/mapping/moving_average_estimation.hpp"
#include "fastdem/mapping/quantile_estimation.hpp"
#include "fastdem/mapping/stat_mean_estimation.hpp"

using namespace fastdem;

// =============================================================================
// Configuration
// =============================================================================

namespace cfg {

constexpr uint32_t RNG_SEED = 42;

// Convergence / dynamic maps: small for fast iteration
constexpr float SMALL_MAP_W = 50.0f;   // [m]
constexpr float SMALL_MAP_H = 50.0f;   // [m]
constexpr float SMALL_MAP_RES = 0.2f;  // [m] → 250×250 = 62.5K cells

// Timing maps: larger to expose per-update cost
constexpr float TIMING_MAP_W = 100.0f;
constexpr float TIMING_MAP_H = 100.0f;
constexpr float TIMING_MAP_RES = 0.1f;  // 1000×1000 = 1M cells

// Convergence benchmark
constexpr int N_SCANS_CONVERGENCE = 100;
const std::vector<float> NOISE_LEVELS = {0.01f, 0.03f, 0.05f, 0.10f};
const std::vector<int> REPORT_SCANS = {1, 5, 10, 25, 50, 100};

// Dynamic response benchmark
constexpr float DYNAMIC_H0 = 0.0f;   // [m] initial height (Phase 1)
constexpr float DYNAMIC_H1 = 0.3f;   // [m] target height (Phase 2)
constexpr int DYNAMIC_PHASE1_SCANS = 50;
constexpr int DYNAMIC_PHASE2_SCANS = 50;
constexpr float DYNAMIC_NOISE = 0.03f;   // [m] measurement noise sigma
constexpr float DYNAMIC_EPSILON = 0.02f; // [m] convergence threshold

// Timing benchmark
const std::vector<size_t> TIMING_OBS_COUNTS = {10000, 30000, 50000, 100000, 200000};
const std::vector<float> TIMING_RESOLUTIONS = {0.05f, 0.1f, 0.2f, 0.5f};
constexpr size_t TIMING_FIXED_OBS = 125000;

// Estimator parameters (defaults used throughout)
constexpr float KALMAN_MIN_VAR = 0.001f;
constexpr float KALMAN_MAX_VAR = 0.1f;
constexpr float KALMAN_PROCESS_NOISE = 0.0f;  // static terrain
constexpr float EMA_ALPHA = 0.5f;             // balanced responsiveness

}  // namespace cfg

// =============================================================================
// Terrain Types
// =============================================================================

enum class TerrainType { FLAT, SLOPE, SINUSOIDAL };

float trueHeight(float x, float y, TerrainType type) {
  switch (type) {
    case TerrainType::FLAT:      return 0.0f;
    case TerrainType::SLOPE:     return 0.01f * x + 0.005f * y;
    case TerrainType::SINUSOIDAL: return 0.3f * std::sin(x * 0.25f) * std::cos(y * 0.25f);
  }
  return 0.0f;
}

const char* terrainName(TerrainType t) {
  switch (t) {
    case TerrainType::FLAT:       return "Flat";
    case TerrainType::SLOPE:      return "Slope";
    case TerrainType::SINUSOIDAL: return "Sinusoidal";
  }
  return "?";
}

// =============================================================================
// Observation Types & Helpers
// =============================================================================

struct Obs {
  grid_map::Index index;
  float z;
  float variance;
};

ElevationMap makeSmallMap() {
  return ElevationMap(cfg::SMALL_MAP_W, cfg::SMALL_MAP_H,
                      cfg::SMALL_MAP_RES, "map");
}

// Iterate over every cell and generate one noisy measurement per cell.
std::vector<Obs> generateFullScan(const ElevationMap& map, TerrainType terrain,
                                  float sigma, std::mt19937& rng) {
  std::normal_distribution<float> noise(0.0f, sigma);
  std::vector<Obs> obs;
  obs.reserve(map.getSize()(0) * map.getSize()(1));

  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
    grid_map::Position pos;
    if (!map.getPosition(*it, pos)) continue;
    float z = trueHeight(pos.x(), pos.y(), terrain) + noise(rng);
    obs.push_back({*it, z, sigma * sigma});
  }
  return obs;
}

// Full-coverage scan at a fixed height (for dynamic response test).
std::vector<Obs> generateFlatScan(const ElevationMap& map, float height,
                                  float sigma, std::mt19937& rng) {
  std::normal_distribution<float> noise(0.0f, sigma);
  std::vector<Obs> obs;
  obs.reserve(map.getSize()(0) * map.getSize()(1));

  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
    obs.push_back({*it, height + noise(rng), sigma * sigma});
  }
  return obs;
}

// Random observations (subset of cells) — used for timing benchmark.
std::vector<Obs> generateTimingObs(const ElevationMap& map, size_t n,
                                   std::mt19937& rng) {
  const auto size = map.getSize();
  std::uniform_int_distribution<int> row_dist(0, size(0) - 1);
  std::uniform_int_distribution<int> col_dist(0, size(1) - 1);
  std::uniform_real_distribution<float> z_dist(-0.5f, 0.5f);

  std::vector<Obs> obs;
  obs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const auto start = map.getStartIndex();
    int r = (row_dist(rng) + start(0)) % size(0);
    int c = (col_dist(rng) + start(1)) % size(1);
    obs.push_back({grid_map::Index(r, c), z_dist(rng), 0.001f});
  }
  return obs;
}

// Compute RMSE over all measured cells against a terrain ground truth.
double computeRMSE(const ElevationMap& map, TerrainType terrain) {
  double sum_sq = 0.0;
  int count = 0;
  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
    float elev = map.elevationAt(*it);
    if (!std::isfinite(elev)) continue;
    grid_map::Position pos;
    if (!map.getPosition(*it, pos)) continue;
    double err = elev - trueHeight(pos.x(), pos.y(), terrain);
    sum_sq += err * err;
    ++count;
  }
  return count > 0 ? std::sqrt(sum_sq / count) : std::numeric_limits<double>::quiet_NaN();
}

// Compute RMSE against a fixed flat height.
double computeRMSEFlat(const ElevationMap& map, float target) {
  double sum_sq = 0.0;
  int count = 0;
  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
    float elev = map.elevationAt(*it);
    if (!std::isfinite(elev)) continue;
    double err = elev - target;
    sum_sq += err * err;
    ++count;
  }
  return count > 0 ? std::sqrt(sum_sq / count) : std::numeric_limits<double>::quiet_NaN();
}

// =============================================================================
// Generic Estimator Helpers
// =============================================================================

// Apply one scan's observations to an estimator (bind + update + computeBounds).
template <typename E>
void applyObs(E& est, ElevationMap& map, const std::vector<Obs>& obs) {
  est.bind(map);
  for (const auto& o : obs) {
    est.update(o.index, o.z, o.variance);
    est.computeBounds(o.index);
  }
}

// =============================================================================
// Section A: Timing Performance
// =============================================================================

template <typename E>
benchmark::Stats timingRun(E est, ElevationMap map,
                           const std::vector<Obs>& obs) {
  est.ensureLayers(map);

  // Warmup (map warms up, caches hot)
  for (int i = 0; i < benchmark::IterationPolicy::WARMUP; ++i) {
    applyObs(est, map, obs);
  }

  return benchmark::runVoid(
      [&]() { applyObs(est, map, obs); },
      benchmark::IterationPolicy::MEDIUM);
}

void runTimingSection() {
  std::cout << "\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << " Section A: Timing Performance\n";
  std::cout << std::string(70, '=') << "\n";

  std::cout << "\nMeasures: bind() + N × (update() + computeBounds())\n";
  std::cout << "Each cell: steady-state (map pre-warmed). Mean over "
            << benchmark::IterationPolicy::MEDIUM << " iterations.\n";
  std::cout << "All times in ms ± 95% CI.\n";

  benchmark::PlatformInfo::capture().print();

  std::mt19937 rng(cfg::RNG_SEED);

  // --- Table 1: Varying observation count ---
  {
    std::cout << "\n--- A1: Varying Observation Count"
              << " (map: " << cfg::TIMING_MAP_W << "x" << cfg::TIMING_MAP_H
              << "m @ " << cfg::TIMING_MAP_RES << "m) ---\n\n";

    ElevationMap base_map(cfg::TIMING_MAP_W, cfg::TIMING_MAP_H,
                          cfg::TIMING_MAP_RES, "map");

    // Pre-generate observations per count
    struct ColData { size_t n; std::vector<Obs> obs; };
    std::vector<ColData> cols;
    for (size_t n : cfg::TIMING_OBS_COUNTS) {
      rng.seed(cfg::RNG_SEED);
      cols.push_back({n, generateTimingObs(base_map, n, rng)});
    }

    // Header
    std::cout << std::left << std::setw(18) << "Estimator";
    for (const auto& col : cols) {
      std::string hdr = std::to_string(col.n / 1000) + "K";
      std::cout << std::right << std::setw(15) << hdr;
    }
    std::cout << "\n" << std::string(18 + 15 * cols.size(), '-') << "\n";

    auto printRow = [&](const std::string& name, auto make_est) {
      std::cout << std::left << std::setw(18) << name;
      for (const auto& col : cols) {
        auto s = timingRun(make_est(), base_map, col.obs);
        std::ostringstream cell;
        cell << std::fixed << std::setprecision(2) << s.mean
             << "±" << std::setprecision(2) << s.ci_95();
        std::cout << std::right << std::setw(15) << cell.str();
      }
      std::cout << "\n";
    };

    printRow("Kalman",
             []() { return Kalman(cfg::KALMAN_MIN_VAR, cfg::KALMAN_MAX_VAR,
                                  cfg::KALMAN_PROCESS_NOISE); });
    printRow("P2Quantile", []() { return P2Quantile(); });
    printRow("StatMean",   []() { return StatMean(); });
    printRow("MovingAvg",  []() { return MovingAverage(cfg::EMA_ALPHA); });

    // Throughput (Mpts/s) for middle obs count
    const size_t mid_n = cfg::TIMING_OBS_COUNTS[cfg::TIMING_OBS_COUNTS.size() / 2];
    const auto& mid_obs = cols[cols.size() / 2].obs;
    std::cout << "\nThroughput at " << mid_n / 1000 << "K observations (Mpts/s):\n";
    auto printThroughput = [&](const std::string& name, auto make_est) {
      auto s = timingRun(make_est(), base_map, mid_obs);
      double tp = mid_n / s.mean / 1000.0;
      std::cout << "  " << std::left << std::setw(16) << name
                << std::fixed << std::setprecision(1) << tp << " Mpts/s\n";
    };
    printThroughput("Kalman",
                    []() { return Kalman(cfg::KALMAN_MIN_VAR, cfg::KALMAN_MAX_VAR,
                                         cfg::KALMAN_PROCESS_NOISE); });
    printThroughput("P2Quantile", []() { return P2Quantile(); });
    printThroughput("StatMean",   []() { return StatMean(); });
    printThroughput("MovingAvg",  []() { return MovingAverage(cfg::EMA_ALPHA); });
  }

  // --- Table 2: Varying resolution ---
  {
    std::cout << "\n--- A2: Varying Resolution (fixed: "
              << cfg::TIMING_FIXED_OBS / 1000 << "K observations) ---\n\n";

    std::cout << std::left << std::setw(18) << "Estimator";
    for (float r : cfg::TIMING_RESOLUTIONS) {
      std::ostringstream hdr;
      hdr << r << "m";
      std::cout << std::right << std::setw(15) << hdr.str();
    }
    std::cout << "\n" << std::string(18 + 15 * cfg::TIMING_RESOLUTIONS.size(), '-') << "\n";

    auto printRow = [&](const std::string& name, auto make_est) {
      std::cout << std::left << std::setw(18) << name;
      for (float res : cfg::TIMING_RESOLUTIONS) {
        rng.seed(cfg::RNG_SEED);
        ElevationMap map(cfg::TIMING_MAP_W, cfg::TIMING_MAP_H, res, "map");
        auto obs = generateTimingObs(map, cfg::TIMING_FIXED_OBS, rng);
        auto s = timingRun(make_est(), map, obs);
        std::ostringstream cell;
        cell << std::fixed << std::setprecision(2) << s.mean
             << "±" << std::setprecision(2) << s.ci_95();
        std::cout << std::right << std::setw(15) << cell.str();
      }
      std::cout << "\n";
    };

    printRow("Kalman",
             []() { return Kalman(cfg::KALMAN_MIN_VAR, cfg::KALMAN_MAX_VAR,
                                  cfg::KALMAN_PROCESS_NOISE); });
    printRow("P2Quantile", []() { return P2Quantile(); });
    printRow("StatMean",   []() { return StatMean(); });
    printRow("MovingAvg",  []() { return MovingAverage(cfg::EMA_ALPHA); });
  }

  std::cout << "\n" << std::string(70, '-') << "\n";
}

// =============================================================================
// Section B: Convergence Accuracy
// =============================================================================

// Runs one complete convergence trial: N scans, returns per-scan RMSE vector.
// Returns NaN entries for scans where the estimator has insufficient data
// (P2Quantile needs >=5 measurements per cell).
template <typename E>
std::vector<double> runConvergenceTrial(E est_proto, TerrainType terrain,
                                        float sigma, std::mt19937& rng) {
  ElevationMap map = makeSmallMap();
  E est = est_proto;
  est.ensureLayers(map);

  std::vector<double> rmse_per_scan;
  rmse_per_scan.reserve(cfg::N_SCANS_CONVERGENCE);

  for (int scan = 0; scan < cfg::N_SCANS_CONVERGENCE; ++scan) {
    auto obs = generateFullScan(map, terrain, sigma, rng);
    applyObs(est, map, obs);
    rmse_per_scan.push_back(computeRMSE(map, terrain));
  }

  return rmse_per_scan;
}

void runConvergenceSection() {
  std::cout << "\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << " Section B: Convergence Accuracy\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << "\nSetup: " << cfg::SMALL_MAP_W << "x" << cfg::SMALL_MAP_H
            << "m map @ " << cfg::SMALL_MAP_RES << "m ("
            << static_cast<int>(cfg::SMALL_MAP_W / cfg::SMALL_MAP_RES) << "x"
            << static_cast<int>(cfg::SMALL_MAP_H / cfg::SMALL_MAP_RES)
            << " cells). Full-coverage scans. RMSE vs ground truth.\n";
  std::cout << "Note: P2Quantile (84th pct) has upward bias ≈ +sigma on zero-mean noise.\n";
  std::cout << "      MovingAverage alpha=" << cfg::EMA_ALPHA << "\n\n";

  // Store full curves for CSV
  // [terrain][noise][estimator][scan]
  struct CurveSet {
    TerrainType terrain;
    float sigma;
    std::string est_name;
    std::vector<double> rmse;
  };
  std::vector<CurveSet> all_curves;

  const std::vector<TerrainType> terrains = {
      TerrainType::FLAT, TerrainType::SINUSOIDAL};

  for (TerrainType terrain : terrains) {
    std::cout << "--- B: Terrain = " << terrainName(terrain) << " ---\n\n";

    // --- Table B1: RMSE at selected scan counts for each noise level ---
    for (float sigma : cfg::NOISE_LEVELS) {
      std::cout << "  Noise sigma=" << std::fixed << std::setprecision(2)
                << sigma << "m\n";
      std::cout << "  " << std::left << std::setw(16) << "Estimator";
      for (int s : cfg::REPORT_SCANS) {
        std::cout << std::right << std::setw(10) << ("scan" + std::to_string(s));
      }
      std::cout << "\n  " << std::string(16 + 10 * cfg::REPORT_SCANS.size(), '-') << "\n";

      auto printConvergenceRow = [&](const std::string& name, auto est_proto) {
        std::mt19937 rng(cfg::RNG_SEED);
        auto rmse_vec = runConvergenceTrial(est_proto, terrain, sigma, rng);

        // Store for CSV
        all_curves.push_back({terrain, sigma, name, rmse_vec});

        std::cout << "  " << std::left << std::setw(16) << name;
        for (int sc : cfg::REPORT_SCANS) {
          double r = rmse_vec[sc - 1];
          if (std::isnan(r)) {
            std::cout << std::right << std::setw(10) << "-";
          } else {
            std::ostringstream cell;
            cell << std::fixed << std::setprecision(4) << r;
            std::cout << std::right << std::setw(10) << cell.str();
          }
        }
        std::cout << "\n";
      };

      printConvergenceRow("Kalman",
                          Kalman(cfg::KALMAN_MIN_VAR, cfg::KALMAN_MAX_VAR,
                                 cfg::KALMAN_PROCESS_NOISE));
      printConvergenceRow("P2Quantile", P2Quantile());
      printConvergenceRow("StatMean",   StatMean());
      printConvergenceRow("MovingAvg",  MovingAverage(cfg::EMA_ALPHA));
      std::cout << "\n";
    }

    // --- Table B2: Steady-state RMSE (last 10 scans avg) across noise levels ---
    std::cout << "  Steady-state RMSE (avg of last 10 scans, scan 91-100):\n";
    std::cout << "  " << std::left << std::setw(16) << "Noise(sigma)";
    for (float s : cfg::NOISE_LEVELS) {
      std::ostringstream hdr;
      hdr << std::fixed << std::setprecision(2) << s << "m";
      std::cout << std::right << std::setw(12) << hdr.str();
    }
    std::cout << "\n  " << std::string(16 + 12 * cfg::NOISE_LEVELS.size(), '-') << "\n";

    struct EstEntry { std::string name; };
    std::vector<std::string> est_names = {"Kalman", "P2Quantile", "StatMean", "MovingAvg"};

    for (const auto& ename : est_names) {
      std::cout << "  " << std::left << std::setw(16) << ename;
      for (float sigma : cfg::NOISE_LEVELS) {
        // Find matching curve
        for (const auto& curve : all_curves) {
          if (curve.terrain == terrain && curve.sigma == sigma &&
              curve.est_name == ename) {
            // Average last 10 scans
            double sum = 0.0;
            int cnt = 0;
            for (int i = 90; i < 100 && i < static_cast<int>(curve.rmse.size()); ++i) {
              if (std::isfinite(curve.rmse[i])) {
                sum += curve.rmse[i];
                ++cnt;
              }
            }
            double avg = (cnt > 0) ? sum / cnt : std::numeric_limits<double>::quiet_NaN();
            if (std::isnan(avg)) {
              std::cout << std::right << std::setw(12) << "-";
            } else {
              std::ostringstream cell;
              cell << std::fixed << std::setprecision(4) << avg;
              std::cout << std::right << std::setw(12) << cell.str();
            }
            break;
          }
        }
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  // --- Write convergence CSVs ---
  std::ofstream csv("convergence_rmse.csv");
  if (csv.is_open()) {
    csv << "terrain,sigma,estimator";
    for (int i = 1; i <= cfg::N_SCANS_CONVERGENCE; ++i) csv << ",scan" << i;
    csv << "\n";
    for (const auto& curve : all_curves) {
      csv << terrainName(curve.terrain) << "," << std::fixed
          << std::setprecision(2) << curve.sigma << "," << curve.est_name;
      for (double r : curve.rmse) {
        if (std::isnan(r))
          csv << ",";
        else
          csv << "," << std::fixed << std::setprecision(6) << r;
      }
      csv << "\n";
    }
    std::cout << "Convergence curves written to: convergence_rmse.csv\n";
  }

  std::cout << std::string(70, '-') << "\n";
}

// =============================================================================
// Section C: Dynamic Response
// =============================================================================

template <typename E>
std::vector<double> runDynamicTrial(E est_proto, float sigma, std::mt19937& rng) {
  ElevationMap map = makeSmallMap();
  E est = est_proto;
  est.ensureLayers(map);

  const int total_scans = cfg::DYNAMIC_PHASE1_SCANS + cfg::DYNAMIC_PHASE2_SCANS;
  std::vector<double> rmse_vec;
  rmse_vec.reserve(total_scans);

  // Phase 1: converge at h0
  for (int s = 0; s < cfg::DYNAMIC_PHASE1_SCANS; ++s) {
    auto obs = generateFlatScan(map, cfg::DYNAMIC_H0, sigma, rng);
    applyObs(est, map, obs);
    rmse_vec.push_back(computeRMSEFlat(map, cfg::DYNAMIC_H0));
  }

  // Phase 2: step to h1
  for (int s = 0; s < cfg::DYNAMIC_PHASE2_SCANS; ++s) {
    auto obs = generateFlatScan(map, cfg::DYNAMIC_H1, sigma, rng);
    applyObs(est, map, obs);
    rmse_vec.push_back(computeRMSEFlat(map, cfg::DYNAMIC_H1));
  }

  return rmse_vec;
}

// Count scans after step change until RMSE drops below epsilon.
int recoveryLatency(const std::vector<double>& rmse_vec, float epsilon) {
  for (int i = cfg::DYNAMIC_PHASE1_SCANS; i < static_cast<int>(rmse_vec.size()); ++i) {
    if (std::isfinite(rmse_vec[i]) && rmse_vec[i] <= epsilon) return i - cfg::DYNAMIC_PHASE1_SCANS + 1;
  }
  return -1;  // did not converge within trial
}

void runDynamicSection() {
  std::cout << "\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << " Section C: Dynamic Response\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << "\nSetup: " << cfg::DYNAMIC_PHASE1_SCANS << " scans at h0="
            << cfg::DYNAMIC_H0 << "m, then " << cfg::DYNAMIC_PHASE2_SCANS
            << " scans at h1=" << cfg::DYNAMIC_H1 << "m.\n";
  std::cout << "Noise sigma=" << cfg::DYNAMIC_NOISE << "m. "
            << "Convergence threshold epsilon=" << cfg::DYNAMIC_EPSILON << "m.\n";
  std::cout << "Recovery latency = scans after step until RMSE <= epsilon.\n\n";

  struct DynamicResult {
    std::string name;
    std::vector<double> rmse;
    int latency;
    double final_rmse;
  };
  std::vector<DynamicResult> results;

  auto addResult = [&](const std::string& name, auto est_proto) {
    std::mt19937 rng(cfg::RNG_SEED);
    auto rmse = runDynamicTrial(est_proto, cfg::DYNAMIC_NOISE, rng);
    int lat = recoveryLatency(rmse, cfg::DYNAMIC_EPSILON);
    double final_r = rmse.back();
    results.push_back({name, rmse, lat, final_r});
  };

  addResult("Kalman",
            Kalman(cfg::KALMAN_MIN_VAR, cfg::KALMAN_MAX_VAR,
                   cfg::KALMAN_PROCESS_NOISE));
  addResult("P2Quantile", P2Quantile());
  addResult("StatMean",   StatMean());
  addResult("MovingAvg",  MovingAverage(cfg::EMA_ALPHA));

  // --- Table C1: Recovery summary ---
  std::cout << std::left << std::setw(18) << "Estimator"
            << std::right << std::setw(20) << "Recovery Latency"
            << std::right << std::setw(20) << "Phase1 RMSE@50"
            << std::right << std::setw(20) << "Final RMSE@100\n";
  std::cout << std::string(78, '-') << "\n";

  for (const auto& r : results) {
    std::string lat_str = (r.latency < 0) ? ">50 scans"
                                          : (std::to_string(r.latency) + " scans");
    double phase1_rmse = r.rmse[cfg::DYNAMIC_PHASE1_SCANS - 1];

    std::cout << std::left << std::setw(18) << r.name
              << std::right << std::setw(20) << lat_str
              << std::right << std::setw(19) << std::fixed << std::setprecision(4)
              << phase1_rmse << "m"
              << std::right << std::setw(19) << r.final_rmse << "m\n";
  }

  // --- Table C2: RMSE at key scan indices after step ---
  std::cout << "\nRMSE vs ground truth h1=" << cfg::DYNAMIC_H1
            << "m after step change (scans post-step):\n";
  std::vector<int> post_step_scans = {1, 2, 3, 5, 10, 20, 50};

  std::cout << std::left << std::setw(18) << "Estimator";
  for (int s : post_step_scans) {
    std::cout << std::right << std::setw(10) << ("+" + std::to_string(s));
  }
  std::cout << "\n" << std::string(18 + 10 * post_step_scans.size(), '-') << "\n";

  for (const auto& r : results) {
    std::cout << std::left << std::setw(18) << r.name;
    for (int s : post_step_scans) {
      int idx = cfg::DYNAMIC_PHASE1_SCANS + s - 1;
      if (idx < static_cast<int>(r.rmse.size())) {
        std::ostringstream cell;
        cell << std::fixed << std::setprecision(4) << r.rmse[idx];
        std::cout << std::right << std::setw(10) << cell.str();
      }
    }
    std::cout << "\n";
  }

  std::cout << "\nNote: Kalman process_noise=" << cfg::KALMAN_PROCESS_NOISE
            << " (static mode). Set process_noise>0 for faster dynamic response.\n";

  // --- Write dynamic CSV ---
  std::ofstream csv("dynamic_response.csv");
  if (csv.is_open()) {
    csv << "scan,phase";
    for (const auto& r : results) csv << "," << r.name << "_rmse";
    csv << "\n";

    const int total = cfg::DYNAMIC_PHASE1_SCANS + cfg::DYNAMIC_PHASE2_SCANS;
    for (int i = 0; i < total; ++i) {
      const char* phase = (i < cfg::DYNAMIC_PHASE1_SCANS) ? "h0" : "h1";
      double ref = (i < cfg::DYNAMIC_PHASE1_SCANS) ? cfg::DYNAMIC_H0 : cfg::DYNAMIC_H1;
      (void)ref;
      csv << (i + 1) << "," << phase;
      for (const auto& r : results) {
        if (i < static_cast<int>(r.rmse.size()))
          csv << "," << std::fixed << std::setprecision(6) << r.rmse[i];
        else
          csv << ",";
      }
      csv << "\n";
    }
    std::cout << "Dynamic response curves written to: dynamic_response.csv\n";
  }

  std::cout << std::string(70, '-') << "\n";
}

// =============================================================================
// Section D: Memory Footprint
// =============================================================================

void runMemorySection() {
  std::cout << "\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << " Section D: Memory Footprint\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << "\nBase layers (always present): elevation, elevation_min, "
               "elevation_max (3 layers)\n";
  std::cout << "Output layers (estimator-created): variance, n_points, "
               "upper_bound, lower_bound\n";
  std::cout << "Internal layers (_prefix): estimator-specific state (not for "
               "visualization)\n\n";

  struct LayerInfo {
    std::string name;
    std::string description;
    bool is_internal;
  };
  struct EstimatorMemory {
    std::string name;
    std::vector<LayerInfo> output_layers;
    std::vector<LayerInfo> internal_layers;
  };

  std::vector<EstimatorMemory> estimators = {
      {"Kalman",
       {{"variance",     "sample variance (Welford)", false},
        {"n_points",     "measurement count",         false},
        {"upper_bound",  "elevation + 2*sigma",       false},
        {"lower_bound",  "elevation - 2*sigma",       false}},
       {{"_kalman_p",    "Kalman filter covariance P",      true},
        {"_sample_mean", "Welford running mean",            true},
        {"_sample_m2",   "Welford M2 accumulator",          true}}},

      {"P2Quantile",
       {{"variance",    "sigma² from (q3-q1)/2",            false},
        {"n_points",    "measurement count",                false},
        {"upper_bound", "99th percentile (q4)",             false},
        {"lower_bound", "1st percentile (q0)",              false}},
       {{"_p2_q[0-4]", "5 P² quantile markers",           true},
        {"_p2_n[0-4]", "5 P² position accumulators",       true}}},

      {"StatMean",
       {{"variance",    "unbiased sample variance",         false},
        {"n_points",    "measurement count",                false},
        {"upper_bound", "elevation + 2*sigma",              false},
        {"lower_bound", "elevation - 2*sigma",              false}},
       {{"_stat_m2",   "Welford M2 accumulator",           true}}},

      {"MovingAverage",
       {{"n_points",    "measurement count",                false},
        {"upper_bound", "= elevation (no variance)",        false},
        {"lower_bound", "= elevation (no variance)",        false}},
       {}},
  };

  // Count extra layers (beyond base elevation, elevation_min, elevation_max)
  // P2Quantile: 10 internal layers (_p2_q0-4, _p2_n0-4)
  std::vector<int> internal_counts = {3, 10, 1, 0};
  std::vector<int> output_counts   = {4,  4, 4, 3};

  const int BYTES_PER_CELL = 4;  // float32
  const int BASE_CELLS_1K  = 1000 * 1000;  // 1000x1000 map

  std::cout << std::left << std::setw(16) << "Estimator"
            << std::right << std::setw(10) << "Out Lyrs"
            << std::right << std::setw(12) << "Int Lyrs"
            << std::right << std::setw(14) << "Total Lyrs"
            << std::right << std::setw(14) << "B/cell"
            << std::right << std::setw(16) << "1K×1K map\n";
  std::cout << std::string(82, '-') << "\n";

  for (size_t i = 0; i < estimators.size(); ++i) {
    int out = output_counts[i];
    int intl = internal_counts[i];
    int total = out + intl;
    int bytes_cell = total * BYTES_PER_CELL;
    double mb_1k = static_cast<double>(BASE_CELLS_1K) * bytes_cell / (1024.0 * 1024.0);

    std::ostringstream mb_str;
    mb_str << std::fixed << std::setprecision(1) << mb_1k << " MB";

    std::cout << std::left << std::setw(16) << estimators[i].name
              << std::right << std::setw(10) << out
              << std::right << std::setw(12) << intl
              << std::right << std::setw(14) << total
              << std::right << std::setw(13) << bytes_cell << "B"
              << std::right << std::setw(16) << mb_str.str() << "\n";
  }

  // Base layer memory for reference
  const int BASE_LAYERS = 3;
  double base_mb = static_cast<double>(BASE_CELLS_1K) * BASE_LAYERS * BYTES_PER_CELL / (1024.0 * 1024.0);
  std::cout << std::string(82, '-') << "\n";
  std::cout << "  Base layers (all estimators): " << BASE_LAYERS
            << " layers = " << std::fixed << std::setprecision(1)
            << base_mb << " MB per 1K×1K map\n\n";

  // Layer details
  for (const auto& est : estimators) {
    std::cout << "  " << est.name << ":\n";
    for (const auto& l : est.output_layers) {
      std::cout << "    [output]   " << std::left << std::setw(16) << l.name
                << " - " << l.description << "\n";
    }
    for (const auto& l : est.internal_layers) {
      std::cout << "    [internal] " << std::left << std::setw(16) << l.name
                << " - " << l.description << "\n";
    }
    std::cout << "\n";
  }

  std::cout << std::string(70, '-') << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
  std::cout << "\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << " FastDEM Estimator Comparison Benchmark\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << "Estimators: Kalman | P2Quantile | StatMean | MovingAverage\n";
  std::cout << "Sections:   A=Timing  B=Convergence  C=Dynamic  D=Memory\n";
  std::cout << std::string(70, '=') << "\n";

  runTimingSection();
  runConvergenceSection();
  runDynamicSection();
  runMemorySection();

  std::cout << "\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << " Done.\n";
  std::cout << "CSV outputs: convergence_rmse.csv, dynamic_response.csv\n";
  std::cout << std::string(70, '=') << "\n\n";

  return 0;
}
