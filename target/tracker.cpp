#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

std::vector<int>
hungarianAlgorithm(const std::vector<std::vector<double>> &costMatrix) {
  int nRows = costMatrix.size();
  if (nRows == 0)
    return std::vector<int>();
  int nCols = costMatrix[0].size();
  int n = std::max(nRows, nCols);
  const double INF = 1e9;

  std::vector<std::vector<double>> a(n, std::vector<double>(n, INF));
  for (int i = 0; i < nRows; i++) {
    for (int j = 0; j < nCols; j++) {
      a[i][j] = costMatrix[i][j];
    }
  }

  std::vector<double> u(n + 1, 0), v(n + 1, 0);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);

  for (int i = 1; i <= n; i++) {
    p[0] = i;
    std::vector<double> minv(n + 1, INF);
    std::vector<bool> used(n + 1, false);
    int j0 = 0;
    do {
      used[j0] = true;
      int i0 = p[j0], j1 = 0;
      double delta = INF;
      for (int j = 1; j <= n; j++) {
        if (!used[j]) {
          double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }
      for (int j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  std::vector<int> assignment(nRows, -1);
  for (int j = 1; j <= n; j++) {
    if (p[j] <= nRows && j <= nCols) {
      assignment[p[j] - 1] = j - 1;
    }
  }
  return assignment;
}

enum class TrackStage { INITIALIZED, CONFIRMED };

struct TrackSettings {
  double measurementNoise;
  double processNoise;
  double covariance;
  double distanceThreshold;
  int maxAge;
  int minHits;
  int maxConsecutiveMisses;
};

class KalmanFilter {
public:
    static constexpr int dimX = 9;
    static constexpr int dimZ = 3;
    
    Eigen::VectorXd x;        // state (9x1)
    Eigen::MatrixXd F;        // state transition matrix (9x9)
    Eigen::MatrixXd H;        // measurement matrix (3x9)
    Eigen::MatrixXd P;        // covariance matrix (9x9)
    Eigen::MatrixXd Q;        // process noise matrix (9x9)
    Eigen::MatrixXd R;        // measurement noise matrix (3x3)
    
    KalmanFilter() {
        x = Eigen::VectorXd::Zero(dimX);
        F = Eigen::MatrixXd::Identity(dimX, dimX);
        H = Eigen::MatrixXd::Zero(dimZ, dimX);
        P = Eigen::MatrixXd::Identity(dimX, dimX);
        Q = Eigen::MatrixXd::Identity(dimX, dimX);
        R = Eigen::MatrixXd::Identity(dimZ, dimZ);
    }
    
    void predict() {
        x = F * x;
        P = F * P * F.transpose() + Q;
    }
    
    void update(const Eigen::VectorXd &z) {
        Eigen::VectorXd y = z - H * x;
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();
        x = x + K * y;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dimX, dimX);
        P = (I - K * H) * P;
    }
};

class Track {
public:
  int id;
  KalmanFilter kalmanFilter;
  TrackStage stage;
  int age;
  int hits;
  int hitStreak;
  int timeSinceUpdate;
  int consecutiveMisses;
  std::vector<Eigen::Vector3d> positionHistory;

  Track(int id, const Eigen::Vector3d &initialPosition,
        const Eigen::Vector3d &initialVelocity,
        const Eigen::Vector3d &initialAcceleration,
        const TrackSettings &settings)
      : id(id), stage(TrackStage::INITIALIZED), age(0), hits(1), hitStreak(0),
        timeSinceUpdate(0), consecutiveMisses(0) {
    initializeKalmanFilter(initialPosition, initialVelocity,
                           initialAcceleration, settings);
    positionHistory.push_back(initialPosition);
  }

  void initializeKalmanFilter(const Eigen::Vector3d &initialPosition,
                              const Eigen::Vector3d &initialVelocity,
                              const Eigen::Vector3d &initialAcceleration,
                              const TrackSettings &settings) {
    kalmanFilter.F =
        Eigen::MatrixXd::Identity(KalmanFilter::dimX, KalmanFilter::dimX);
    kalmanFilter.F(0, 3) = 1.0;
    kalmanFilter.F(0, 6) = 0.5;
    kalmanFilter.F(1, 4) = 1.0;
    kalmanFilter.F(1, 7) = 0.5;
    kalmanFilter.F(2, 5) = 1.0;
    kalmanFilter.F(2, 8) = 0.5;
    kalmanFilter.F(3, 6) = 1.0;
    kalmanFilter.F(4, 7) = 1.0;
    kalmanFilter.F(5, 8) = 1.0;

    kalmanFilter.H =
        Eigen::MatrixXd::Zero(KalmanFilter::dimZ, KalmanFilter::dimX);
    kalmanFilter.H(0, 0) = 1.0;
    kalmanFilter.H(1, 1) = 1.0;
    kalmanFilter.H(2, 2) = 1.0;

    kalmanFilter.R *= settings.measurementNoise;
    kalmanFilter.P *= settings.covariance;
    kalmanFilter.Q *= settings.processNoise;

    kalmanFilter.x.segment(0, 3) = initialPosition;
    kalmanFilter.x.segment(3, 3) = initialVelocity;
    kalmanFilter.x.segment(6, 3) = initialAcceleration;
  }

  Eigen::VectorXd predict() {
    kalmanFilter.predict();
    age++;
    consecutiveMisses++;
    return kalmanFilter.x;
  }

  void update(const Eigen::Vector3d &measurement) {
    Eigen::VectorXd z(KalmanFilter::dimZ);
    z << measurement(0), measurement(1), measurement(2);
    kalmanFilter.update(z);
    timeSinceUpdate = 0;
    hits++;
    hitStreak++;
    consecutiveMisses = 0;
    positionHistory.push_back(measurement);
    if (positionHistory.size() > 5) {
      positionHistory.erase(positionHistory.begin());
    }
  }

  Eigen::Vector3d getState() const { return kalmanFilter.x.segment(0, 3); }

  Eigen::Vector3d getVelocity() const { return kalmanFilter.x.segment(3, 3); }

  Eigen::Vector3d getAcceleration() const {
    return kalmanFilter.x.segment(6, 3);
  }

  Eigen::Vector3d getSmoothedPosition() const {
    Eigen::Vector3d mean(0, 0, 0);
    for (const auto &pos : positionHistory) {
      mean += pos;
    }
    if (!positionHistory.empty()) {
      mean /= positionHistory.size();
    }
    return mean;
  }

  std::string repr() const {
    std::ostringstream oss;
    oss << "Track " << id << ": " << getState().transpose()
        << " | Velocity: " << getVelocity().transpose()
        << " | Acceleration: " << getAcceleration().transpose() << " | Stage: "
        << (stage == TrackStage::CONFIRMED ? "CONFIRMED" : "INITIALIZED")
        << " | Age: " << age << " | Hits: " << hits
        << " | HitStreak: " << hitStreak
        << " | TimeSinceUpdate: " << timeSinceUpdate
        << " | ConsecutiveMisses: " << consecutiveMisses;
    return oss.str();
  }
};

class Tracker {
public:
  std::vector<Track> tracks;
  int trackId;
  double distanceThreshold;
  int maxAge;
  int minHits;
  int maxConsecutiveMisses;
  TrackSettings settings;

  Tracker(const TrackSettings &settings)
      : trackId(0), distanceThreshold(settings.distanceThreshold),
        maxAge(settings.maxAge), minHits(settings.minHits),
        maxConsecutiveMisses(settings.maxConsecutiveMisses),
        settings(settings) {}

  void
  associateDetectionsToTracks(const std::vector<Eigen::Vector3d> &detections,
                              std::vector<int> &assignedTracks,
                              std::vector<int> &unassignedTracks,
                              std::vector<int> &unassignedDetections) {
    assignedTracks.clear();
    unassignedTracks.clear();
    unassignedDetections.clear();

    if (tracks.empty()) {
      for (int i = 0; i < detections.size(); i++)
        unassignedDetections.push_back(i);
      return;
    }

    std::vector<std::vector<double>> costMatrix;
    for (size_t i = 0; i < tracks.size(); i++) {
      Eigen::Vector3d predicted = tracks[i].getState();
      std::vector<double> costRow;
      for (size_t j = 0; j < detections.size(); j++) {
        double dist = (predicted - detections[j]).norm();
        costRow.push_back(dist);
      }
      costMatrix.push_back(costRow);
    }

    std::vector<int> assignments = hungarianAlgorithm(costMatrix);
    std::vector<bool> detectionAssigned(detections.size(), false);

    for (size_t i = 0; i < assignments.size(); i++) {
      int detectionIndex = assignments[i];
      if (detectionIndex != -1 && detectionIndex < costMatrix[i].size() &&
          costMatrix[i][detectionIndex] < distanceThreshold) {
        tracks[i].update(detections[detectionIndex]);
        assignedTracks.push_back(static_cast<int>(i));
        detectionAssigned[detectionIndex] = true;
      }
    }

    for (size_t i = 0; i < tracks.size(); i++) {
      if (std::find(assignedTracks.begin(), assignedTracks.end(),
                    static_cast<int>(i)) == assignedTracks.end())
        unassignedTracks.push_back(static_cast<int>(i));
    }

    for (size_t j = 0; j < detections.size(); j++) {
      if (!detectionAssigned[j])
        unassignedDetections.push_back(static_cast<int>(j));
    }
  }

  void predictTracks() {
    for (auto &track : tracks) {
      track.predict();
    }
  }

  void updateTracks(const std::vector<Eigen::Vector3d> &detections) {
    std::vector<int> assignedTracks;
    std::vector<int> unassignedTracks;
    std::vector<int> unassignedDetections;

    associateDetectionsToTracks(detections, assignedTracks, unassignedTracks,
                                unassignedDetections);

    for (int index : unassignedDetections) {
      Eigen::Vector3d initPosition = detections[index];
      Eigen::Vector3d initVelocity = Eigen::Vector3d::Zero();
      Eigen::Vector3d initAcceleration = Eigen::Vector3d::Zero();
      tracks.emplace_back(trackId, initPosition, initVelocity, initAcceleration,
                          settings);
      trackId++;
    }

    std::sort(unassignedTracks.begin(), unassignedTracks.end(),
              std::greater<int>());
    for (int i : unassignedTracks) {
      tracks[i].timeSinceUpdate++;
      if (tracks[i].timeSinceUpdate > maxAge ||
          tracks[i].consecutiveMisses > maxConsecutiveMisses) {
        tracks.erase(tracks.begin() + i);
      }
    }

    for (auto &track : tracks) {
      if (track.hits >= minHits && track.stage == TrackStage::INITIALIZED)
        track.stage = TrackStage::CONFIRMED;
      if (track.timeSinceUpdate > 1)
        track.hitStreak = 0;
    }
  }

  std::vector<Track> &getTracks() { return tracks; }
};

json runTrackerWithParameters(const TrackSettings &trackerSettings,
                              const json &detectionsJson) {
  Tracker tracker(trackerSettings);
  json outputData = json::object();

  std::vector<int> frameNumbers;
  std::map<int, std::string> frameNumberToKey;

  for (const auto &el : detectionsJson.items()) {
    int frame = std::stoi(el.key());
    frameNumbers.push_back(frame);
    frameNumberToKey[frame] = el.key();
  }
  std::sort(frameNumbers.begin(), frameNumbers.end());

  for (int frame : frameNumbers) {
    std::string frameKey = frameNumberToKey[frame];
    const json &content = detectionsJson[frameKey];

    std::vector<Eigen::Vector3d> frameDetections;
    for (const auto &obj : content["tracks"]) {
      double x = obj["x"];
      double y = obj["y"];
      double z = obj["z"];
      frameDetections.emplace_back(x, y, z);
    }

    tracker.predictTracks();
    tracker.updateTracks(frameDetections);

    std::cout << "Frame " << frameKey << " tracks:" << std::endl;
    for (const auto &track : tracker.getTracks()) {
      std::cout << track.repr() << std::endl;
    }

    json tracksArray = json::array();
    for (const auto &track : tracker.getTracks()) {
      if (track.stage == TrackStage::CONFIRMED) {
        Eigen::Vector3d smoothed = track.getSmoothedPosition();
        Eigen::Vector3d velocity = track.getVelocity();
        Eigen::Vector3d acceleration = track.getAcceleration();

        json trackJson = json::object();
        trackJson["id"] = track.id;
        trackJson["x"] = smoothed(0);
        trackJson["y"] = smoothed(1);
        trackJson["z"] = smoothed(2);
        trackJson["vx"] = velocity(0);
        trackJson["vy"] = velocity(1);
        trackJson["vz"] = velocity(2);
        trackJson["ax"] = acceleration(0);
        trackJson["ay"] = acceleration(1);
        trackJson["az"] = acceleration(2);

        tracksArray.push_back(trackJson);
      }
    }
    outputData[frameKey] = {{"tracks", tracksArray}};
  }
  return outputData;
}

json loadJson(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Unable to open: " << filepath << std::endl;
    exit(1);
  }
  json j;
  file >> j;
  return j;
}

void saveJson(const std::string &filepath, const json &data) {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Unable to open for writing: " << filepath << std::endl;
    exit(1);
  }
  file << data.dump(4);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
      std::cerr << "Usage: " << argv[0] << " <parametersPath> <detectionsPath> <trackedPath>" << std::endl;
      return 1;
  }

  std::string parametersPath = argv[1];
  std::string detectionsPath = argv[2];
  std::string trackedPath = argv[3];

  json detections = loadJson(detectionsPath);
  json parameters = loadJson(parametersPath);

  TrackSettings trackerSettings = {parameters["measurement_noise"],
                                   parameters["process_noise"],
                                   parameters["covariance"],
                                   parameters["distance_threshold"],
                                   parameters["max_age"],
                                   parameters["min_hits"],
                                   parameters["max_consecutive_misses"]};

  json outputData = runTrackerWithParameters(trackerSettings, detections);

  saveJson(trackedPath, outputData);

  std::cout << "Tracking complete; results written to " << trackedPath
            << std::endl;

  return 0;
}
