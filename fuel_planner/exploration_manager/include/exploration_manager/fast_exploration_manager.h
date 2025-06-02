#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>

using Eigen::Vector3d;
using std::shared_ptr;
using std::vector;

namespace fast_planner {

class EDTEnvironment;
class SDFMap;
class FastPlannerManager;
class FrontierFinder;
struct ExplorationParam;
struct ExplorationData;

enum EXPL_RESULT { NO_FRONTIER, FAIL, SUCCEED };

class FastExplorationManager {
public:
  FastExplorationManager();
  ~FastExplorationManager();

  void initialize(ros::NodeHandle& nh);

  int planExploreMotion(const Vector3d& pos,
                        const Vector3d& vel,
                        const Vector3d& acc,
                        const Vector3d& yaw);

  int classicFrontier(const Vector3d& pos, double yaw);
  int rapidFrontier(const Vector3d& pos,
                    const Vector3d& vel,
                    double yaw,
                    bool& classic);

  shared_ptr<ExplorationData> ed_;
  shared_ptr<ExplorationParam> ep_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<FrontierFinder> frontier_finder_;

private:
  // Environment and map
  shared_ptr<EDTEnvironment> edt_environment_;
  shared_ptr<SDFMap> sdf_map_;

  // Hysteresis & switch penalty
  bool has_last_target_ = false;
  Vector3d last_target_;       
  double hysteresis_dist_ = 1.0;
  double switch_penalty_ = 2.0;

  // GPR seed-only mode
  vector<Vector3d> seed_points_;

  // Dynamic cube (GPR) mode control
  bool dynamic_cube_active_ = false;
  ros::Subscriber exploration_ing_sub_;  // /exploration_ing

  // Exploration end control
  bool exploration_end_ = false;
  ros::Subscriber exploration_end_sub_;  // /exploration_end

  // GPR peak seeds
  vector<Vector3d> gpr_peaks_;

  // Callbacks
  void explorationIngCallback(const std_msgs::Bool::ConstPtr& msg);
  void explorationEndCallback(const std_msgs::Bool::ConstPtr& msg);

  // Core planning methods
  void findGlobalTour(const Vector3d& cur_pos,
                      const Vector3d& cur_vel,
                      Vector3d cur_yaw,
                      vector<int>& indices);

  void refineLocalTour(const Vector3d& cur_pos,
                       const Vector3d& cur_vel,
                       const Vector3d& cur_yaw,
                       const vector<vector<Vector3d>>& n_points,
                       const vector<vector<double>>& n_yaws,
                       vector<Vector3d>& refined_pts,
                       vector<double>& refined_yaws);

  void shortenPath(vector<Vector3d>& path);

public:
  typedef shared_ptr<FastExplorationManager> Ptr;
};

}  // namespace fast_planner

#endif  // _EXPLORATION_MANAGER_H_
