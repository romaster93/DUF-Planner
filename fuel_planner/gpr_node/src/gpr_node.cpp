#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include </home/cho/catkin_ws/devel/include/gazebo_radiation_plugins/Simulated_Radiation_Msg.h>
// frontier_finder 헤더
#include </home/cho/catkin_ws/src/MARSIM/fuel_planner/active_perception/include/active_perception/frontier_finder.h>
// EDTEnvironment 헤더
#include </home/cho/catkin_ws/src/MARSIM/fuel_planner/plan_env/include/plan_env/edt_environment.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "gpr.h"
#include <vector>
#include <numeric>
#include <cmath>
#include <std_msgs/Bool.h> 

using Vec3 = Eigen::Vector3d;

// GPR 객체 (length_scale, nu, alpha, noise_level)
GPR gpr(1.5, 1.0, 0.01, 1.0);

// 상태 변수
bool position_received   = false;
Vec3 current_position;
bool orientation_received = false;
Eigen::Quaterniond current_orientation;
std_msgs::Bool exploration_start;
std_msgs::Bool exploration_stop;

// dynamic cube
bool dynamic_cube_active   = false;
Vec3 dynamic_cube_center;
Eigen::Matrix3d dynamic_cube_rotation;
const double dynamic_cube_threshold = 20.0;
const double grid_resolution        = 0.5;
bool manual_next_cube = false;
bool manual_end = false;
Vec3  last_stop_center    = Vec3::Zero();
double last_stop_radius   = std::sqrt(1.5*1.5*2 + 3.0*3.0);  // ≈3.674m
const double stop_margin  = 1.0;  // 외벽 + 1m


// 퍼블리셔
ros::Publisher heatmap_pub;
ros::Publisher uncertainty_pub;
ros::Publisher exploration_start_pub;
ros::Publisher exploration_end_pub;

// 마지막으로 퍼블리시한 불확실도 저장
static std::vector<double> last_var;

// 수렴 판단용 임계값 (μ_σ - σ_σ) 한 번만 계산
static bool    threshold_initialized = false;
static double  sigma_threshold       = 0.0;
const double   converge_fraction     = 0.6;  // 80%

// Peak 이동 제어
static bool has_last_peak = false;
static Vec3 last_peak;
const double SAME_PEAK_THRESH = 0.3;  // m 단위

// —— Helper 함수들 ——

// 1) 큐브 내 전역 격자점 생성
std::vector<Vec3> generateGridPoints() {
    std::vector<Vec3> pts;
    if (!dynamic_cube_active) return pts;
    for (double x = -1.5; x <= 1.5; x += grid_resolution)
    for (double y =  -1.5; y <= 1.5; y += grid_resolution)
    for (double z = -1.5; z <= 1.5; z += grid_resolution) {
        Vec3 local(x,y,z);
        pts.push_back(dynamic_cube_center + dynamic_cube_rotation * local);
    }
    return pts;
}


void nextCubeTriggerCallback(const std_msgs::Bool::ConstPtr& data) {
    if (data->data) {
      manual_next_cube = true;
      ROS_WARN("Manual next-cube trigger received.");
    }
}
  
void endTriggerCallback(const std_msgs::Bool::ConstPtr& data) {
    if (data->data) {
      manual_end = true;
      ROS_WARN("Manual exploration-end trigger received.");
    }
}

// 2) PointCloud2로 퍼블리시 (intensity 필드에 values)
void publishPointCloud(const std::vector<Vec3>& pts,
                       const std::vector<double>& values,
                       ros::Publisher& pub)
{
    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.header.frame_id = "camera_init_sec";
    cloud.header.stamp    = pcl_conversions::toPCL(ros::Time::now());
    cloud.points.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        pcl::PointXYZI p;
        p.x = pts[i].x(); p.y = pts[i].y(); p.z = pts[i].z();
        p.intensity = static_cast<float>(values[i]);
        cloud.points.push_back(p);
    }
    cloud.width  = cloud.points.size();
    cloud.height = 1;
    sensor_msgs::PointCloud2 out;
    pcl::toROSMsg(cloud, out);
    out.header.frame_id = cloud.header.frame_id;
    pub.publish(out);
}

// 3) Pose 콜백: 현재 위치/오리엔테이션 갱신
void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_position = Vec3(msg->pose.position.x,
                            msg->pose.position.y,
                            msg->pose.position.z);
    position_received    = true;
    current_orientation  = Eigen::Quaterniond(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z);
    orientation_received = true;
}

// 4) Radiation 콜백: GPR 학습, 예측, 그리고 “Peak 이동” 제어
// void radiationCallback(const gazebo_radiation_plugins::Simulated_Radiation_Msg::ConstPtr& msg) {
//     if (!position_received) return;

//     if (manual_end) {
//         manual_end = false;
//         if (dynamic_cube_active) {
//           dynamic_cube_active = false;
//           // 정지 위치 기록
//           last_stop_center = dynamic_cube_center;
//           ROS_WARN("Exploration forcefully ended by manual trigger.");
//         }
//         return;
//     }

//     // (1) 학습 데이터 축적
//     gpr.addTrainingPoint(current_position, msg->value);

//     // (2) dynamic cube 최초 활성화
//     if (msg->value >= dynamic_cube_threshold && !dynamic_cube_active) {
//         dynamic_cube_active   = true;
//         dynamic_cube_center   = current_position;
//         dynamic_cube_rotation = orientation_received
//             ? current_orientation.toRotationMatrix()
//             : Eigen::Matrix3d::Identity();
//         ROS_INFO("Dynamic cube activated at [%.2f,%.2f,%.2f].",
//                  dynamic_cube_center.x(),
//                  dynamic_cube_center.y(),
//                  dynamic_cube_center.z());
//     }
//     if (!dynamic_cube_active) return;

//     // (3) GPR 학습 & 예측
//     gpr.fitModel();
//     auto grid = generateGridPoints();
//     int n = grid.size();
//     if (n == 0) return;

//     Eigen::MatrixXd X(n,3);
//     for (int i = 0; i < n; ++i) X.row(i) = grid[i];

//     Eigen::VectorXd mu, sigma;
//     gpr.predictBatchWithUncertainty(X, mu, sigma);

//     // mean / var 벡터 복사
//     std::vector<double> mean(mu.data(),   mu.data()+n);
//     std::vector<double> var (sigma.data(),sigma.data()+n);

//     // (4) Heatmap 항상 퍼블리시
//     publishPointCloud(grid, mean, heatmap_pub);
//     // ROS_INFO("Published heatmap.");

//     // (5) Uncertainty 은 변경 시에만 퍼블리시
//     bool publish_unc = false;
//     if (last_var.size() != var.size()) {
//         publish_unc = true;
//     } else {
//         for (size_t i = 0; i < var.size(); ++i) {
//             if (std::fabs(var[i] - last_var[i]) > 1e-6) {
//                 publish_unc = true;
//                 break;
//             }
//         }
//     }
//     if (publish_unc) {
//         publishPointCloud(grid, var, uncertainty_pub);
//         last_var = var;
//         // ROS_INFO("Published uncertainty_map.");
//     }

//     // (6) Threshold 한 번만 계산 (mean_σ - std_σ)
//     if (!threshold_initialized) {
//         double sum = std::accumulate(var.begin(), var.end(), 0.0);
//         double mean_sigma = sum / double(n);
//         double sq_sum = 0.0;
//         for (double s : var) sq_sum += (s - mean_sigma)*(s - mean_sigma);
//         double std_sigma = std::sqrt(sq_sum / double(n));
//         sigma_threshold = mean_sigma - std_sigma;
//         threshold_initialized = true;
//         ROS_INFO("Initialized sigma_threshold = %.4f (mean %.4f - std %.4f)",
//                  sigma_threshold, mean_sigma, std_sigma);
//     }

//     // (7) 수렴 검사: σ ≤ σ_thresh 인 비율 ≥ 80% 이면 “Peak 이동” 단계로
//     int low_count = 0;
//     for (double s : var) if (s <= sigma_threshold) ++low_count;
//     double frac = double(low_count) / double(n);
//     ROS_INFO("Convergence: %d/%d = %.1f%% of points σ ≤ %.4f",low_count, n, frac * 100.0, sigma_threshold);
//     if (frac >= converge_fraction || manual_trigger) {

//         if (manual_trigger) {
//             manual_trigger = false;
//             ROS_WARN("Manual trigger: forcing next cube.");
//         }

//         // (8) 현재 mean field 에서 최대 heat intensity 위치 찾기
//         int idx_max = std::distance(
//             mean.begin(),
//             std::max_element(mean.begin(), mean.end())
//         );
//         Vec3 next_peak = grid[idx_max];

//         if (!has_last_peak) {
//             // 최초 Peak
//             last_peak       = next_peak;
//             has_last_peak   = true;
//             dynamic_cube_center = next_peak;
//             threshold_initialized = false;
//             last_var.clear();
//             ROS_WARN("First peak at [%.2f,%.2f,%.2f] → moving cube.",
//                      next_peak.x(), next_peak.y(), next_peak.z());
//         }
//         else {
//             // 이전 Peak 과 비교
//             double dist = (next_peak - last_peak).norm();
//             if (dist < SAME_PEAK_THRESH) {
//                 // 같은 봉우리로 수렴 → 탐색 종료
//                 ROS_WARN("Peak stabilized at [%.2f,%.2f,%.2f] (dist=%.2fm) → done.",
//                          next_peak.x(), next_peak.y(), next_peak.z(), dist);
//                 dynamic_cube_active = false;
//             } else {
//                 // 새로운 Peak → 이동 재시작
//                 last_peak       = next_peak;
//                 dynamic_cube_center = next_peak;
//                 threshold_initialized = false;
//                 last_var.clear();
//                 ROS_WARN("New peak at [%.2f,%.2f,%.2f] (dist=%.2fm) → moving cube.",
//                          next_peak.x(), next_peak.y(), next_peak.z(), dist);
//             }
//         }
//     }
// }

void radiationCallback(const gazebo_radiation_plugins::Simulated_Radiation_Msg::ConstPtr& msg) {
    if (!position_received) return;
  
    // ─────────────────────────────────────────────────────────
    // 1) 수동 탐색 종료 플래그 처리
    if (manual_end) {
      manual_end = false;
      if (dynamic_cube_active) {
        dynamic_cube_active = false;
        exploration_stop.data = true;
        exploration_end_pub.publish(exploration_stop);

        // 정지 위치 기록
        last_stop_center = dynamic_cube_center;
        ROS_WARN("Exploration forcefully ended by manual trigger.");
      }
      return;
    }
  
    // ─────────────────────────────────────────────────────────
    // 2) 아직 활성화 안 된 상태 → 시작 조건 검사
    if (!dynamic_cube_active) {
      // (수동 다음 큐브 트리거)
      if (manual_next_cube) {
        manual_next_cube = false;
        // 다만, 이전 정지 위치 반경 + margin 이내면 무시
        double d = (current_position - last_stop_center).norm();
        if (d > last_stop_radius + stop_margin) {
          dynamic_cube_active   = true;
          exploration_start.data = dynamic_cube_active;
          exploration_start_pub.publish(exploration_start);
          dynamic_cube_center   = current_position;
          dynamic_cube_rotation = orientation_received
              ? current_orientation.toRotationMatrix()
              : Eigen::Matrix3d::Identity();
          threshold_initialized = false;
          last_var.clear();
          ROS_WARN("Manual next-cube trigger → new cube at [%.2f,%.2f,%.2f]",
                   dynamic_cube_center.x(),
                   dynamic_cube_center.y(),
                   dynamic_cube_center.z());
        }
      }
      // (자동 시작: threshold 초과 + 이전 정지 반경 바깥)
      else if (msg->value >= dynamic_cube_threshold) {
        double d = (current_position - last_stop_center).norm();
        if (d > last_stop_radius + stop_margin) {
          dynamic_cube_active   = true;
          exploration_start.data = dynamic_cube_active;
          exploration_start_pub.publish(exploration_start);
          dynamic_cube_center   = current_position;
          dynamic_cube_rotation = orientation_received
              ? current_orientation.toRotationMatrix()
              : Eigen::Matrix3d::Identity();
          threshold_initialized = false;
          last_var.clear();
          ROS_INFO("Dynamic cube auto-activated at [%.2f,%.2f,%.2f].",
                   dynamic_cube_center.x(),
                   dynamic_cube_center.y(),
                   dynamic_cube_center.z());
        }
      }
      return;
    }
  
    // ─────────────────────────────────────────────────────────
    // 3) 이미 활성화된 상태 → GPR 학습/예측 & “다음 큐브” 조건 검사
  
    // (1) 학습 데이터 누적
    gpr.addTrainingPoint(current_position, msg->value);
  
    // (2) 모델 학습 & 예측
    gpr.fitModel();
    auto grid = generateGridPoints();
    int n = grid.size();
    if (n == 0) return;
  
    Eigen::MatrixXd X(n,3);
    for (int i = 0; i < n; ++i) X.row(i) = grid[i];
    Eigen::VectorXd mu, sigma;
    gpr.predictBatchWithUncertainty(X, mu, sigma);
  
    // (3) 시각화 퍼블리시
    std::vector<double> mean(mu.data(),   mu.data()+n);
    std::vector<double> var (sigma.data(),sigma.data()+n);
    publishPointCloud(grid, mean, heatmap_pub);
    bool publish_unc = (last_var.size() != var.size());
    if (!publish_unc) {
      for (size_t i = 0; i < var.size(); ++i)
        if (std::fabs(var[i] - last_var[i]) > 1e-6) { publish_unc = true; break; }
    }
    if (publish_unc) {
      publishPointCloud(grid, var, uncertainty_pub);
      last_var = var;
    }
  
    // (4) 수렴 검사: σ ≤ σ_thresh 인 비율 ≥ converge_fraction
    if (!threshold_initialized) {
      double sum = std::accumulate(var.begin(), var.end(), 0.0);
      double mean_sigma = sum / double(n);
      double sq_sum = 0.0;
      for (double s : var) sq_sum += (s - mean_sigma)*(s - mean_sigma);
      double std_sigma = std::sqrt(sq_sum / double(n));
      sigma_threshold = mean_sigma - std_sigma;
      threshold_initialized = true;
      ROS_INFO("Initialized sigma_threshold = %.4f", sigma_threshold);
    }
    int low_count = std::count_if(var.begin(), var.end(), [&](double s){ return s <= sigma_threshold; });
    double frac = double(low_count)/double(n);
    ROS_INFO("Convergence: %.1f%% σ ≤ %.4f", frac*100.0, sigma_threshold);
  
    // ─────────────────────────────────────────────────────────
    // 5) “다음 큐브” 트리거(자동 or 수동)
    if (frac >= converge_fraction || manual_next_cube) {
      manual_next_cube = false;
  
      int idx_max = std::distance(mean.begin(), std::max_element(mean.begin(), mean.end()));
      Vec3 next_peak = grid[idx_max];
  
      // 새 큐브로 이동
      last_peak             = next_peak;
      dynamic_cube_center   = next_peak;
      threshold_initialized = false;
      last_var.clear();
      ROS_WARN("Moving to next cube at [%.2f,%.2f,%.2f] (%.1f%% converged%s).",
               next_peak.x(), next_peak.y(), next_peak.z(),
               frac*100.0,
               frac>=converge_fraction?"":" + manual trigger");
    }
  }
  

int main(int argc, char** argv) {
    ros::init(argc, argv, "gpr_node");
    ros::NodeHandle nh;

    heatmap_pub     = nh.advertise<sensor_msgs::PointCloud2>("/heatmap",        1, true);
    uncertainty_pub = nh.advertise<sensor_msgs::PointCloud2>("/uncertainty_map",1, true);
    exploration_start_pub = nh.advertise<std_msgs::Bool>("/exploration_ing", 1);
    exploration_end_pub = nh.advertise<std_msgs::Bool>("/exploration_end", 1);

    ros::Subscriber ps = nh.subscribe("/mavros/local_position/pose",      1, poseCallback);
    ros::Subscriber rs = nh.subscribe("/radiation_sensor_plugin/sensor_1",1, radiationCallback);
    ros::Subscriber next_trig = nh.subscribe("/next_cube_trigger", 1, nextCubeTriggerCallback);
    ros::Subscriber end_trig = nh.subscribe("/exploration_end_trigger", 1, endTriggerCallback);

    ROS_INFO("GPR ROS node started. Real-time heatmap & uncertainty → dynamic cube peaks.");
    ros::spin();
    return 0;
}
