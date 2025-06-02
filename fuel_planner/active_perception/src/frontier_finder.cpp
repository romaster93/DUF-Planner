#include <active_perception/frontier_finder.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>
// #include <path_searching/astar2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/edt_environment.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_node.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl_ros/point_cloud.h> 

// use PCL region growing segmentation
// #include <pcl/point_types.h>
// #include <pcl/search/search.h>
// #include <pcl/search/kdtree.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/segmentation/region_growing.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>

#include <Eigen/Eigenvalues>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

namespace fast_planner {
FrontierFinder::FrontierFinder(const EDTEnvironment::Ptr& edt, ros::NodeHandle& nh) {
  this->edt_env_ = edt;
  int voxel_num = edt->sdf_map_->getVoxelNum();
  frontier_flag_ = vector<char>(voxel_num, 0);
  fill(frontier_flag_.begin(), frontier_flag_.end(), 0);

  nh.param("frontier/cluster_min", cluster_min_, -1);
  nh.param("frontier/cluster_size_xy", cluster_size_xy_, -1.0);
  nh.param("frontier/cluster_size_z", cluster_size_z_, -1.0);
  nh.param("frontier/min_candidate_dist", min_candidate_dist_, -1.0);
  nh.param("frontier/min_candidate_clearance", min_candidate_clearance_, -1.0);
  nh.param("frontier/candidate_dphi", candidate_dphi_, -1.0);
  nh.param("frontier/candidate_rmax", candidate_rmax_, -1.0);
  nh.param("frontier/candidate_rmin", candidate_rmin_, -1.0);
  nh.param("frontier/candidate_rnum", candidate_rnum_, -1);
  nh.param("frontier/down_sample", down_sample_, -1);
  nh.param("frontier/min_visib_num", min_visib_num_, -1);
  nh.param("frontier/min_view_finish_fraction", min_view_finish_fraction_, -1.0);

  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  percep_utils_.reset(new PerceptionUtils(nh));

  pose_sub_ = nh.subscribe("/mavros/local_position/pose", 1,
                             &FrontierFinder::poseCallback, this);
  rad_sub_  = nh.subscribe("/radiation_sensor_plugin/sensor_1", 1,
                             &FrontierFinder::radiationCallback, this);
  uncertainty_field_sub_ =  nh.subscribe("/uncertainty_map", 1, &FrontierFinder::uncertaintyCallback, this);
  estimation_field_sub_ = nh.subscribe("/heatmap", 1, &FrontierFinder::estimationCallback, this);

  uncertainty_seeds_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/uncertainty_seeds", 1, true);
}

FrontierFinder::~FrontierFinder() {
}

void FrontierFinder::searchFrontiers() {
  ros::Time t1 = ros::Time::now();
  tmp_frontiers_.clear();

  // Bounding box of updated region
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max, true);

  // Removed changed frontiers in updated map
  auto resetFlag = [&](list<Frontier>::iterator& iter, list<Frontier>& frontiers) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {
      edt_env_->sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toadr(idx)] = 0;
    }
    iter = frontiers.erase(iter);
  };

  std::cout << "Before remove: " << frontiers_.size() << std::endl;

  removed_ids_.clear();
  int rmv_idx = 0;
  for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter)) {
      resetFlag(iter, frontiers_);
      removed_ids_.push_back(rmv_idx);
    } else {
      ++rmv_idx;
      ++iter;
    }
  }
  std::cout << "After remove: " << frontiers_.size() << std::endl;
  for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter))
      resetFlag(iter, dormant_frontiers_);
    else
      ++iter;
  }
  
  // ★ 불확실도 기반 시드 추가 ★
  // uncertaintyCallback() 에서 모인 uncandidate_pts_ 를
  // expandFrontier() 호출로 tmp_frontiers_ 에 추가합니다.
  for (const auto& pt : uncandidate_pts_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(pt, idx);
    int adr = toadr(idx);
    // 아직 클러스터에 포함되지 않았고, free + 인접 unknown 이면
    if (frontier_flag_[adr] == 0 && knownfree(idx)){ // && isNeighborUnknown(idx)) {
      ROS_INFO("PRINT IDX!!!!!!!");
      ROS_INFO_STREAM(idx);
      expandFrontier(idx);
    }
  }

  // Search new frontier within box slightly inflated from updated box
  Vector3d search_min = update_min - Vector3d(1, 1, 0.5);
  Vector3d search_max = update_max + Vector3d(1, 1, 0.5);
  Vector3d box_min, box_max;
  edt_env_->sdf_map_->getBox(box_min, box_max);
  for (int k = 0; k < 3; ++k) {
    search_min[k] = max(search_min[k], box_min[k]);
    search_max[k] = min(search_max[k], box_max[k]);
  }
  Eigen::Vector3i min_id, max_id;
  edt_env_->sdf_map_->posToIndex(search_min, min_id);
  edt_env_->sdf_map_->posToIndex(search_max, max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        // Scanning the updated region to find seeds of frontiers
        Eigen::Vector3i cur(x, y, z);
        if (frontier_flag_[toadr(cur)] == 0 && knownfree(cur) && isNeighborUnknown(cur)) {
          // Expand from the seed cell to find a complete frontier cluster
          expandFrontier(cur);
        }
      }
  splitLargeFrontiers(tmp_frontiers_);

  ROS_WARN_THROTTLE(5.0, "Frontier t: %lf", (ros::Time::now() - t1).toSec());
}

void FrontierFinder::expandFrontier(
    const Eigen::Vector3i& first /* , const int& depth, const int& parent_id */) {
  // std::cout << "depth: " << depth << std::endl;
  auto t1 = ros::Time::now();

  // Data for clustering
  queue<Eigen::Vector3i> cell_queue;
  vector<Eigen::Vector3d> expanded;
  Vector3d pos;

  edt_env_->sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);
  cell_queue.push(first);
  frontier_flag_[toadr(first)] = 1;

  // Search frontier cluster based on region growing (distance clustering)
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);
    for (auto nbr : nbrs) {
      // Qualified cell should be inside bounding box and frontier cell not clustered
      int adr = toadr(nbr);
      if (frontier_flag_[adr] == 1 || !edt_env_->sdf_map_->isInBox(nbr) ||
          !(knownfree(nbr) && isNeighborUnknown(nbr)))
        continue;

      edt_env_->sdf_map_->indexToPos(nbr, pos);
      if (pos[2] < 0.4) continue;  // Remove noise close to ground
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = 1;
    }
  }
  if (expanded.size() > cluster_min_) {
    // Compute detailed info
    Frontier frontier;
    frontier.cells_ = expanded;
    computeFrontierInfo(frontier);
    tmp_frontiers_.push_back(frontier);
  }
}

void FrontierFinder::splitLargeFrontiers(list<Frontier>& frontiers) {
  list<Frontier> splits, tmps;
  for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
    // Check if each frontier needs to be split horizontally
    if (splitHorizontally(*it, splits)) {
      tmps.insert(tmps.end(), splits.begin(), splits.end());
      splits.clear();
    } else
      tmps.push_back(*it);
  }
  frontiers = tmps;
}

bool FrontierFinder::splitHorizontally(const Frontier& frontier, list<Frontier>& splits) {
  // Split a frontier into small piece if it is too large
  auto mean = frontier.average_.head<2>();
  bool need_split = false;
  for (auto cell : frontier.filtered_cells_) {
    if ((cell.head<2>() - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }
  if (!need_split) return false;

  // Compute principal component
  // Covariance matrix of cells
  Eigen::Matrix2d cov;
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    Eigen::Vector2d diff = cell.head<2>() - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  // Find eigenvector corresponds to maximal eigenvector
  Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector2d first_pc = vectors.col(max_idx);
  std::cout << "max idx: " << max_idx << std::endl;
  std::cout << "mean: " << mean.transpose() << ", first pc: " << first_pc.transpose() << std::endl;

  // Split the frontier into two groups along the first PC
  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    if ((cell.head<2>() - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // Recursive call to split frontier that is still too large
  list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::isInBoxes(
    const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx) {
  Vector3d pt;
  edt_env_->sdf_map_->indexToPos(idx, pt);
  for (auto box : boxes) {
    // Check if contained by a box
    bool inbox = true;
    for (int i = 0; i < 3; ++i) {
      inbox = inbox && pt[i] > box.first[i] && pt[i] < box.second[i];
      if (!inbox) break;
    }
    if (inbox) return true;
  }
  return false;
}

void FrontierFinder::updateFrontierCostMatrix() {
  std::cout << "cost mat size before remove: " << std::endl;
  for (auto ftr : frontiers_)
    std::cout << "(" << ftr.costs_.size() << "," << ftr.paths_.size() << "), ";
  std::cout << "" << std::endl;

  std::cout << "cost mat size remove: " << std::endl;
  if (!removed_ids_.empty()) {
    // 내림차순으로 정리
    std::sort(removed_ids_.rbegin(), removed_ids_.rend());

    std::cout << "Removing indices: ";
    for (int idx : removed_ids_) std::cout << idx << " ";
    std::cout << std::endl;
    for (auto it = frontiers_.begin(); it != first_new_ftr_; ++it) {
      for (int rem_idx : removed_ids_) {
        // bounds 체크
        if (rem_idx < 0) continue;
        if (rem_idx < static_cast<int>(it->costs_.size())) {
          auto cost_it = std::next(it->costs_.begin(), rem_idx);
          it->costs_.erase(cost_it);
        }
        if (rem_idx < static_cast<int>(it->paths_.size())) {
          auto path_it = std::next(it->paths_.begin(), rem_idx);
          it->paths_.erase(path_it);
        }
      }
      std::cout << "(" << it->costs_.size() << "," << it->paths_.size() << "), ";
    }
    std::cout << std::endl;

    removed_ids_.clear();
  }
  //   // Delete path and cost for removed clusters
  //   for (auto it = frontiers_.begin(); it != first_new_ftr_; ++it) {
  //     auto cost_iter = it->costs_.begin();
  //     auto path_iter = it->paths_.begin();
  //     int iter_idx = 0;
  //     for (int i = 0; i < removed_ids_.size(); ++i) {
  //       // Step iterator to the item to be removed
  //       while (iter_idx < removed_ids_[i]) {
  //         ++cost_iter;
  //         ++path_iter;
  //         ++iter_idx;
  //       }
  //       cost_iter = it->costs_.erase(cost_iter);
  //       path_iter = it->paths_.erase(path_iter);
  //     }
  //     std::cout << "(" << it->costs_.size() << "," << it->paths_.size() << "), ";
  //   }
  //   removed_ids_.clear();
  // }
  // std::cout << "" << std::endl;

  auto updateCost = [](const list<Frontier>::iterator& it1, const list<Frontier>::iterator& it2) {
    std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
    // Search path from old cluster's top viewpoint to new cluster'
    Viewpoint& vui = it1->viewpoints_.front();
    Viewpoint& vuj = it2->viewpoints_.front();
    vector<Vector3d> path_ij;
    double cost_ij = ViewNode::computeCost(
        vui.pos_, vuj.pos_, vui.yaw_, vuj.yaw_, Vector3d(0, 0, 0), 0, path_ij);
    // Insert item for both old and new clusters
    it1->costs_.push_back(cost_ij);
    it1->paths_.push_back(path_ij);
    reverse(path_ij.begin(), path_ij.end());
    it2->costs_.push_back(cost_ij);
    it2->paths_.push_back(path_ij);
  };

  std::cout << "cost mat add: " << std::endl;
  // Compute path and cost between old and new clusters
  for (auto it1 = frontiers_.begin(); it1 != first_new_ftr_; ++it1)
    for (auto it2 = first_new_ftr_; it2 != frontiers_.end(); ++it2)
      updateCost(it1, it2);

  // Compute path and cost between new clusters
  for (auto it1 = first_new_ftr_; it1 != frontiers_.end(); ++it1)
    for (auto it2 = it1; it2 != frontiers_.end(); ++it2) {
      if (it1 == it2) {
        std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
        it1->costs_.push_back(0);
        it1->paths_.push_back({});
      } else
        updateCost(it1, it2);
    }
  std::cout << "" << std::endl;
  std::cout << "cost mat size final: " << std::endl;
  for (auto ftr : frontiers_)
    std::cout << "(" << ftr.costs_.size() << "," << ftr.paths_.size() << "), ";
  std::cout << "" << std::endl;
}

void FrontierFinder::mergeFrontiers(Frontier& ftr1, const Frontier& ftr2) {
  // Merge ftr2 into ftr1
  ftr1.average_ =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  ftr1.cells_.insert(ftr1.cells_.end(), ftr2.cells_.begin(), ftr2.cells_.end());
  computeFrontierInfo(ftr1);
}

bool FrontierFinder::canBeMerged(const Frontier& ftr1, const Frontier& ftr2) {
  Vector3d merged_avg =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  // Check if it can merge two frontier without exceeding size limit
  for (auto c1 : ftr1.cells_) {
    auto diff = c1 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  for (auto c2 : ftr2.cells_) {
    auto diff = c2 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  return true;
}

bool FrontierFinder::haveOverlap(
    const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2) {
  // Check if two box have overlap part
  Vector3d bmin, bmax;
  for (int i = 0; i < 3; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3) return false;
  }
  return true;
}

bool FrontierFinder::isFrontierChanged(const Frontier& ft) {
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && isNeighborUnknown(idx))) return true;
  }
  return false;
}

void FrontierFinder::computeFrontierInfo(Frontier& ftr) {
  // Compute average position and bounding box of cluster
  ftr.average_.setZero();
  ftr.box_max_ = ftr.cells_.front();
  ftr.box_min_ = ftr.cells_.front();
  for (auto cell : ftr.cells_) {
    ftr.average_ += cell;
    for (int i = 0; i < 3; ++i) {
      ftr.box_min_[i] = min(ftr.box_min_[i], cell[i]);
      ftr.box_max_[i] = max(ftr.box_max_[i], cell[i]);
    }
  }
  ftr.average_ /= double(ftr.cells_.size());

  // Compute downsampled cluster
  downsample(ftr.cells_, ftr.filtered_cells_);
}

void FrontierFinder::computeFrontiersToVisit() {
  first_new_ftr_ = frontiers_.end();
  int new_num = 0;
  int new_dormant_num = 0;
  // Try find viewpoints for each cluster and categorize them according to viewpoint number
  for (auto& tmp_ftr : tmp_frontiers_) {
    // Search viewpoints around frontier
    sampleViewpoints(tmp_ftr);
    if (!tmp_ftr.viewpoints_.empty()) {
      ++new_num;
      list<Frontier>::iterator inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);
      // Sort the viewpoints by coverage fraction, best view in front
      sort(
          inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
          [](const Viewpoint& v1, const Viewpoint& v2) { return v1.visib_num_ > v2.visib_num_; });
      if (first_new_ftr_ == frontiers_.end()) first_new_ftr_ = inserted;
    } else {
      // Find no viewpoint, move cluster to dormant list
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }
  // Reset indices of frontiers
  int idx = 0;
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
    std::cout << ft.id_ << ", ";
  }
  std::cout << "\nnew num: " << new_num << ", new dormant: " << new_dormant_num << std::endl;
  std::cout << "to visit: " << frontiers_.size() << ", dormant: " << dormant_frontiers_.size()
            << std::endl;
}

void FrontierFinder::getTopViewpointsInfo(
    const Vector3d& cur_pos, vector<Eigen::Vector3d>& points, vector<double>& yaws,
    vector<Eigen::Vector3d>& averages) {
  points.clear();
  yaws.clear();
  averages.clear();
  for (auto frontier : frontiers_) {
    bool no_view = true;
    for (auto view : frontier.viewpoints_) {
      // Retrieve the first viewpoint that is far enough and has highest coverage
      if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
      points.push_back(view.pos_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
      no_view = false;
      break;
    }
    if (no_view) {
      // All viewpoints are very close, just use the first one (with highest coverage).
      auto view = frontier.viewpoints_.front();
      points.push_back(view.pos_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
    }
  }
}

void FrontierFinder::getViewpointsInfo(
    const Vector3d& cur_pos, const vector<int>& ids, const int& view_num, const double& max_decay,
    vector<vector<Eigen::Vector3d>>& points, vector<vector<double>>& yaws) {
  points.clear();
  yaws.clear();
  for (auto id : ids) {
    // Scan all frontiers to find one with the same id
    for (auto frontier : frontiers_) {
      if (frontier.id_ == id) {
        // Get several top viewpoints that are far enough
        vector<Eigen::Vector3d> pts;
        vector<double> ys;
        int visib_thresh = frontier.viewpoints_.front().visib_num_ * max_decay;
        for (auto view : frontier.viewpoints_) {
          if (pts.size() >= view_num || view.visib_num_ <= visib_thresh) break;
          if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
          pts.push_back(view.pos_);
          ys.push_back(view.yaw_);
        }
        if (pts.empty()) {
          // All viewpoints are very close, ignore the distance limit
          for (auto view : frontier.viewpoints_) {
            if (pts.size() >= view_num || view.visib_num_ <= visib_thresh) break;
            pts.push_back(view.pos_);
            ys.push_back(view.yaw_);
          }
        }
        points.push_back(pts);
        yaws.push_back(ys);
      }
    }
  }
}

void FrontierFinder::getFrontiers(vector<vector<Eigen::Vector3d>>& clusters) {
  clusters.clear();
  for (auto frontier : frontiers_)
    clusters.push_back(frontier.cells_);
  // clusters.push_back(frontier.filtered_cells_);
}

void FrontierFinder::getDormantFrontiers(vector<vector<Vector3d>>& clusters) {
  clusters.clear();
  for (auto ft : dormant_frontiers_)
    clusters.push_back(ft.cells_);
}

void FrontierFinder::getFrontierBoxes(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes) {
  boxes.clear();
  for (auto frontier : frontiers_) {
    Vector3d center = (frontier.box_max_ + frontier.box_min_) * 0.5;
    Vector3d scale = frontier.box_max_ - frontier.box_min_;
    boxes.push_back(make_pair(center, scale));
  }
}

// void FrontierFinder::getPathForTour(
//     const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path) {
//   // Make an frontier_indexer to access the frontier list easier
//   vector<list<Frontier>::iterator> frontier_indexer;
//   for (auto it = frontiers_.begin(); it != frontiers_.end(); ++it)
//     frontier_indexer.push_back(it);

//   // Compute the path from current pos to the first frontier
//   vector<Vector3d> segment;
//   ViewNode::searchPath(pos, frontier_indexer[frontier_ids[0]]->viewpoints_.front().pos_, segment);
//   path.insert(path.end(), segment.begin(), segment.end());

//   // Get paths of tour passing all clusters
//   for (int i = 0; i < frontier_ids.size() - 1; ++i) {
//     // Move to path to next cluster
//     auto path_iter = frontier_indexer[frontier_ids[i]]->paths_.begin();
//     int next_idx = frontier_ids[i + 1];
//     for (int j = 0; j < next_idx; ++j)
//       ++path_iter;
//     path.insert(path.end(), path_iter->begin(), path_iter->end());
//   }
// }

//위에가 원본함수, 아래는 메모리 폭발 방지 장치 추가한 코드
void FrontierFinder::getPathForTour(
  const Vector3d& pos, 
  const vector<int>& frontier_ids, 
  vector<Vector3d>& path)
{
// 1) frontier_indexer 만들기
vector<list<Frontier>::iterator> frontier_indexer;
for (auto it = frontiers_.begin(); it != frontiers_.end(); ++it)
  frontier_indexer.push_back(it);

if (frontier_ids.empty()) {
  ROS_WARN("getPathForTour: frontier_ids 가 비어있습니다.");
  return;
}
int N = frontier_indexer.size();

// 2) 첫 cluster 로 가는 path
path.clear();
vector<Vector3d> segment;
int first_id = frontier_ids[0];
if (first_id < 0 || first_id >= N) {
  ROS_ERROR("getPathForTour: 첫 ID(%d) 가 범위를 벗어남 [0,%d)", first_id, N);
} else {
  ViewNode::searchPath(
    pos,
    frontier_indexer[first_id]->viewpoints_.front().pos_,
    segment
  );
  path.insert(path.end(), segment.begin(), segment.end());
}

// 3) tour 중간 구간들
for (size_t i = 0; i + 1 < frontier_ids.size(); ++i) {
  int cur_id  = frontier_ids[i];
  int next_id = frontier_ids[i+1];
  // 안전 검사
  if (cur_id < 0 || cur_id >= N || next_id < 0 || next_id >= N) {
    ROS_ERROR("getPathForTour: ID 범위 오류 cur=%d,next=%d (N=%d)", 
              cur_id, next_id, N);
    continue;
  }

  // 이터레이터 잡고
  auto& paths_list = frontier_indexer[cur_id]->paths_;
  if (paths_list.empty()) {
    ROS_WARN("getPathForTour: frontier %d 의 paths_ 가 비어있습니다.", cur_id);
    continue;
  }

  // next_id 만큼 advance 하되, 넘으면 끝으로 clamp
  auto path_iter = paths_list.begin();
  int max_j = std::min(next_id, int(paths_list.size()) - 1);
  std::advance(path_iter, max_j);

  // 삽입
  path.insert(path.end(),
              path_iter->begin(),
              path_iter->end());
}
}

// void FrontierFinder::getFullCostMatrix(
//     const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
//     Eigen::MatrixXd& mat) {
//   if (false) {
//     // Use symmetric TSP formulation
//     int dim = frontiers_.size() + 2;
//     mat.resize(dim, dim);  // current pose (0), sites, and virtual depot finally

//     int i = 1, j = 1;
//     for (auto ftr : frontiers_) {
//       for (auto cs : ftr.costs_)
//         mat(i, j++) = cs;
//       ++i;
//       j = 1;
//     }

//     // Costs from current pose to sites
//     for (auto ftr : frontiers_) {
//       Viewpoint vj = ftr.viewpoints_.front();
//       vector<Vector3d> path;
//       mat(0, j) = mat(j, 0) =
//           ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path);
//       ++j;
//     }
//     // Costs from depot to sites, the same large vaule
//     for (j = 1; j < dim - 1; ++j) {
//       mat(dim - 1, j) = mat(j, dim - 1) = 100;
//     }
//     // Zero cost to depot to ensure connection
//     mat(0, dim - 1) = mat(dim - 1, 0) = -10000;

//   } else {
//     // Use Asymmetric TSP
//     int dimen = frontiers_.size();
//     mat.resize(dimen + 1, dimen + 1);
//     // std::cout << "mat size: " << mat.rows() << ", " << mat.cols() << std::endl;
//     // Fill block for clusters
//     int i = 1, j = 1;
//     for (auto ftr : frontiers_) {
//       for (auto cs : ftr.costs_) {
//         // std::cout << "(" << i << ", " << j << ")"
//         // << ", ";
//         mat(i, j++) = cs;
//       }
//       ++i;
//       j = 1;
//     }
//     // std::cout << "" << std::endl;

//     // Fill block from current state to clusters
//     mat.leftCols<1>().setZero();
//     for (auto ftr : frontiers_) {
//       // std::cout << "(0, " << j << ")"
//       // << ", ";
//       Viewpoint vj = ftr.viewpoints_.front();
//       vector<Vector3d> path;
//       mat(0, j++) =
//           ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path);
//     }
//     // std::cout << "" << std::endl;
//   }
// }

// 위 함수가 원본 함수임.
// frontier_finder.cpp 에서 이 부분만 통째로 교체하세요.
void FrontierFinder::getFullCostMatrix(
  const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
  Eigen::MatrixXd& mat) {
int N = int(frontiers_.size());
// mat 은 (0: 현재 위치, 1~N: 각 frontier) 으로 쓸 거니까 (N+1)x(N+1)
mat.resize(N+1, N+1);

// 1) frontier ↔ frontier 비용 채우기
auto it_i = frontiers_.begin();
for (int i = 0; i < N; ++i, ++it_i) {
  auto it_j = frontiers_.begin();
  for (int j = 0; j < N; ++j, ++it_j) {
    if (i == j) {
      mat(i+1, j+1) = 0.0;
    } else {
      const Viewpoint& vi = it_i->viewpoints_.front();
      const Viewpoint& vj = it_j->viewpoints_.front();
      // 중간 경로는 필요 없으니 빈 벡터로
      std::vector<Vector3d> dummy_path;
      double cost = ViewNode::computeCost(
        vi.pos_, vj.pos_,
        vi.yaw_, vj.yaw_,
        Vector3d(0,0,0), 0,
        dummy_path
      );
      mat(i+1, j+1) = cost;
    }
  }
}

// 2) 현재 위치 → 각 frontier 비용 채우기
mat(0,0) = 0.0;
int col = 1;
for (auto& ftr : frontiers_) {
  const Viewpoint& vj = ftr.viewpoints_.front();
  std::vector<Vector3d> dummy_path;
  double cost = ViewNode::computeCost(
    cur_pos, vj.pos_,
    cur_yaw[0], vj.yaw_,
    cur_vel, cur_yaw[1],
    dummy_path
  );
  mat(0, col++) = cost;
}
}

void FrontierFinder::findViewpoints(
    const Vector3d& sample, const Vector3d& ftr_avg, vector<Viewpoint>& vps) {
  if (!edt_env_->sdf_map_->isInBox(sample) ||
      edt_env_->sdf_map_->getInflateOccupancy(sample) == 1 || isNearUnknown(sample))
    return;

  double left_angle_, right_angle_, vertical_angle_, ray_length_;

  // Central yaw is determined by frontier's average position and sample
  auto dir = ftr_avg - sample;
  double hc = atan2(dir[1], dir[0]);

  vector<int> slice_gains;
  // Evaluate info gain of different slices
  for (double phi_h = -M_PI_2; phi_h <= M_PI_2 + 1e-3; phi_h += M_PI / 18) {
    // Compute gain of one slice
    int gain = 0;
    for (double phi_v = -vertical_angle_; phi_v <= vertical_angle_; phi_v += vertical_angle_ / 3) {
      // Find endpoint of a ray
      Vector3d end;
      end[0] = sample[0] + ray_length_ * cos(phi_v) * cos(hc + phi_h);
      end[1] = sample[1] + ray_length_ * cos(phi_v) * sin(hc + phi_h);
      end[2] = sample[2] + ray_length_ * sin(phi_v);

      // Do raycasting to check info gain
      Vector3i idx;
      raycaster_->input(sample, end);
      while (raycaster_->nextId(idx)) {
        // Hit obstacle, stop the ray
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 || !edt_env_->sdf_map_->isInBox(idx))
          break;
        // Count number of unknown cells
        if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) ++gain;
      }
    }
    slice_gains.push_back(gain);
  }

  // Sum up slices' gain to get different yaw's gain
  vector<pair<double, int>> yaw_gains;
  for (int i = 0; i < 6; ++i)  // [-90,-10]-> [10,90], delta_yaw = 20, 6 groups
  {
    double yaw = hc - M_PI_2 + M_PI / 9.0 * i + right_angle_;
    int gain = 0;
    for (int j = 2 * i; j < 2 * i + 9; ++j)  // 80 degree hFOV, 9 slices
      gain += slice_gains[j];
    yaw_gains.push_back(make_pair(yaw, gain));
  }

  // Get several yaws with highest gain
  vps.clear();
  sort(
      yaw_gains.begin(), yaw_gains.end(),
      [](const pair<double, int>& p1, const pair<double, int>& p2) {
        return p1.second > p2.second;
      });
  for (int i = 0; i < 3; ++i) {
    if (yaw_gains[i].second < min_visib_num_) break;
    Viewpoint vp = { sample, yaw_gains[i].first, yaw_gains[i].second };
    while (vp.yaw_ < -M_PI)
      vp.yaw_ += 2 * M_PI;
    while (vp.yaw_ > M_PI)
      vp.yaw_ -= 2 * M_PI;
    vps.push_back(vp);
  }
}

// Sample viewpoints around frontier's average position, check coverage to the frontier cells
void FrontierFinder::sampleViewpoints(Frontier& frontier) {
  // Evaluate sample viewpoints on circles, find ones that cover most cells
  for (double rc = candidate_rmin_, dr = (candidate_rmax_ - candidate_rmin_) / candidate_rnum_;
       rc <= candidate_rmax_ + 1e-3; rc += dr)
    for (double phi = -M_PI; phi < M_PI; phi += candidate_dphi_) {
      const Vector3d sample_pos = frontier.average_ + rc * Vector3d(cos(phi), sin(phi), 0);

      // Qualified viewpoint is in bounding box and in safe region
      if (!edt_env_->sdf_map_->isInBox(sample_pos) ||
          edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 || isNearUnknown(sample_pos))
        continue;

      // Compute average yaw
      auto& cells = frontier.filtered_cells_;
      Eigen::Vector3d ref_dir = (cells.front() - sample_pos).normalized();
      double avg_yaw = 0.0;
      for (int i = 1; i < cells.size(); ++i) {
        Eigen::Vector3d dir = (cells[i] - sample_pos).normalized();
        double yaw = acos(dir.dot(ref_dir));
        if (ref_dir.cross(dir)[2] < 0) yaw = -yaw;
        avg_yaw += yaw;
      }
      avg_yaw = avg_yaw / cells.size() + atan2(ref_dir[1], ref_dir[0]);
      wrapYaw(avg_yaw);
      // Compute the fraction of covered and visible cells
      int visib_num = countVisibleCells(sample_pos, avg_yaw, cells);
      if (visib_num > min_visib_num_) {
        Viewpoint vp = { sample_pos, avg_yaw, visib_num };
        frontier.viewpoints_.push_back(vp);
        // int gain = findMaxGainYaw(sample_pos, frontier, sample_yaw);
      }
      // }
    }
}

bool FrontierFinder::isFrontierCovered() {
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max);

  auto checkChanges = [&](const list<Frontier>& frontiers) {
    for (auto ftr : frontiers) {
      if (!haveOverlap(ftr.box_min_, ftr.box_max_, update_min, update_max)) continue;
      const int change_thresh = min_view_finish_fraction_ * ftr.cells_.size();
      int change_num = 0;
      for (auto cell : ftr.cells_) {
        Eigen::Vector3i idx;
        edt_env_->sdf_map_->posToIndex(cell, idx);
        if (!(knownfree(idx) && isNeighborUnknown(idx)) && ++change_num >= change_thresh)
          return true;
      }
    }
    return false;
  };

  if (checkChanges(frontiers_) || checkChanges(dormant_frontiers_)) return true;

  return false;
}

bool FrontierFinder::isNearUnknown(const Eigen::Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -1; z <= 1; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

int FrontierFinder::countVisibleCells(
    const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& cluster) {
  percep_utils_->setPose(pos, yaw);
  int visib_num = 0;
  Eigen::Vector3i idx;
  for (auto cell : cluster) {
    // Check if frontier cell is inside FOV
    if (!percep_utils_->insideFOV(cell)) continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->input(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (visib) visib_num += 1;
  }
  return visib_num;
}

void FrontierFinder::downsample(
    const vector<Eigen::Vector3d>& cluster_in, vector<Eigen::Vector3d>& cluster_out) {
  // downsamping cluster
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudf(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto cell : cluster_in)
    cloud->points.emplace_back(cell[0], cell[1], cell[2]);

  const double leaf_size = edt_env_->sdf_map_->getResolution() * down_sample_;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloudf);

  cluster_out.clear();
  for (auto pt : cloudf->points)
    cluster_out.emplace_back(pt.x, pt.y, pt.z);
}

void FrontierFinder::wrapYaw(double& yaw) {
  while (yaw < -M_PI)
    yaw += 2 * M_PI;
  while (yaw > M_PI)
    yaw -= 2 * M_PI;
}

void FrontierFinder::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  current_pose_ = *msg;
}

// Radiation 콜백: current_radiation_ 에 저장만 합니다.
void FrontierFinder::radiationCallback(const gazebo_radiation_plugins::Simulated_Radiation_Msg::ConstPtr& msg) {
   current_radiation_ = msg->value;
  //  ROS_INFO("RADIATION CALLBACK ENABLE!!!!");
}

// // 불확실성 콜백 구현
// void FrontierFinder::uncertaintyCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
//   uncandidate_pts_.clear();

//   // sensor_msgs::PointCloud2Iterator 로 x,y,z,intensity 읽기
//   sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
//   sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
//   sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");
//   sensor_msgs::PointCloud2ConstIterator<float> it_int(*msg, "intensity");

//   for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z, ++it_int) {
//     float unc = *it_int;
//     if (unc > uncertainty_threshold_) {
//       Eigen::Vector3d pt(*it_x, *it_y, *it_z);
//       uncandidate_pts_.push_back(pt);
//     }
//   }
//   ROS_INFO("UNCERTAINTY CALLBACK PLAY!!!!!!!!!!");
  
//   ROS_INFO("uncertaintyCallback: selected point amount = %zu", uncandidate_pts_.size());

//   pcl::PointCloud<pcl::PointXYZ> cloud;
//   cloud.header.frame_id = msg->header.frame_id;
//   cloud.height = 1;
//   cloud.width  = uncandidate_pts_.size();
//   cloud.points.reserve(uncandidate_pts_.size());
//   for (const auto& pt : uncandidate_pts_) {
//     cloud.points.emplace_back(pt.x(), pt.y(), pt.z());
//   }

//   sensor_msgs::PointCloud2 out;
//   pcl::toROSMsg(cloud, out);

//   out.header = msg->header;

//   uncertainty_seeds_pub_.publish(out);

//   ROS_INFO("uncertaintyCallback: selected point amount = %zu", uncandidate_pts_.size());

// }

void FrontierFinder::uncertaintyCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  // 1) raw 후보점 추출
  std::vector<Eigen::Vector3d> raw_pts;
  sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg,"x"),
                                        it_y(*msg,"y"),
                                        it_z(*msg,"z"),
                                        it_int(*msg,"intensity");
  for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z, ++it_int) {
    if (*it_int > uncertainty_threshold_) {
      raw_pts.emplace_back(*it_x, *it_y, *it_z);
    }
  }
  ROS_INFO("uncertaintyCallback: raw candidates = %zu", raw_pts.size());

  // 2) PCL 클라우드로 변환
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->header   = pcl_conversions::toPCL(msg->header);
  cloud->height   = 1;
  cloud->width    = raw_pts.size();
  cloud->points.reserve(raw_pts.size());
  for (auto& p : raw_pts)
    cloud->points.emplace_back(p.x(), p.y(), p.z());

  // 3) VoxelGrid 다운샘플링 (균일 분포 확보)
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vox(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  float leaf_size = 0.5f;  // 환경 해상도나 원하는 간격에 맞춰 조절
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.filter(*cloud_vox);
  ROS_INFO(" after VoxelGrid: %zu → %zu", cloud->size(), cloud_vox->size());

  // 4) OpenCV k-means 으로 최대 6개 클러스터 추출
  int N = cloud_vox->size();
  int K = std::min(N, 30);
  if (N == 0) {
    ROS_WARN("VoxelGrid 후 남은 포인트가 없습니다.");
    return;
  }
  cv::Mat samples(N, 3, CV_32F);
  for (int i = 0; i < N; ++i) {
    samples.at<float>(i, 0) = cloud_vox->points[i].x;
    samples.at<float>(i, 1) = cloud_vox->points[i].y;
    samples.at<float>(i, 2) = cloud_vox->points[i].z;
  }
  cv::Mat labels, centers;
  cv::kmeans(samples, K, labels,
             cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 100, 1e-4),
             3, cv::KMEANS_PP_CENTERS, centers);

  // 5) uncandidate_pts_ 에 클러스터 중심을 저장
  uncandidate_pts_.clear();
  uncandidate_pts_.reserve(K);
  pcl::PointCloud<pcl::PointXYZ> cloud_out;
  cloud_out.header.frame_id = msg->header.frame_id;
  cloud_out.height = 1;
  cloud_out.width  = K;
  cloud_out.points.reserve(K);
  for (int k = 0; k < K; ++k) {
    Eigen::Vector3d c(
      centers.at<float>(k,0),
      centers.at<float>(k,1),
      centers.at<float>(k,2)
    );
    uncandidate_pts_.push_back(c);
    cloud_out.points.emplace_back(c.x(), c.y(), c.z());
  }
  ROS_INFO("uncertaintyCallback: clustered to %zu seeds", uncandidate_pts_.size());

  // 6) PointCloud2 로 퍼블리시
  sensor_msgs::PointCloud2 out;
  pcl::toROSMsg(cloud_out, out);
  out.header = msg->header;
  uncertainty_seeds_pub_.publish(out);
}

void FrontierFinder::estimationCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  sensor_msgs::PointCloud2 test_cloud2;
  test_cloud2 = *msg; 
  ROS_INFO("GPR POINTCLOUD CALLBACK PLAY!!!!!!!!!!");
}


Eigen::Vector3i FrontierFinder::searchClearVoxel(const Eigen::Vector3i& pt) {
  queue<Eigen::Vector3i> init_que;
  vector<Eigen::Vector3i> nbrs;
  Eigen::Vector3i cur, start_idx;
  init_que.push(pt);
  // visited_flag_[toadr(pt)] = 1;

  while (!init_que.empty()) {
    cur = init_que.front();
    init_que.pop();
    if (knownfree(cur)) {
      start_idx = cur;
      break;
    }

    nbrs = sixNeighbors(cur);
    for (auto nbr : nbrs) {
      int adr = toadr(nbr);
      // if (visited_flag_[adr] == 0)
      // {
      //   init_que.push(nbr);
      //   visited_flag_[adr] = 1;
      // }
    }
  }
  return start_idx;
}

inline vector<Eigen::Vector3i> FrontierFinder::sixNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(6);
  Eigen::Vector3i tmp;

  tmp = voxel - Eigen::Vector3i(1, 0, 0);
  neighbors[0] = tmp;
  tmp = voxel + Eigen::Vector3i(1, 0, 0);
  neighbors[1] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 1, 0);
  neighbors[2] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 1, 0);
  neighbors[3] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 0, 1);
  neighbors[4] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 0, 1);
  neighbors[5] = tmp;

  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::tenNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(10);
  Eigen::Vector3i tmp;
  int count = 0;

  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0) continue;
      tmp = voxel + Eigen::Vector3i(x, y, 0);
      neighbors[count++] = tmp;
    }
  }
  neighbors[count++] = tmp - Eigen::Vector3i(0, 0, 1);
  neighbors[count++] = tmp + Eigen::Vector3i(0, 0, 1);
  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::allNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(26);
  Eigen::Vector3i tmp;
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) continue;
        tmp = voxel + Eigen::Vector3i(x, y, z);
        neighbors[count++] = tmp;
      }
  return neighbors;
}

inline bool FrontierFinder::isNeighborUnknown(const Eigen::Vector3i& voxel) {
  // At least one neighbor is unknown
  auto nbrs = sixNeighbors(voxel);
  for (auto nbr : nbrs) {
    if (edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::UNKNOWN) return true;
  }
  return false;
}

inline int FrontierFinder::toadr(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->toAddress(idx);
}

inline bool FrontierFinder::knownfree(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::FREE;
}

inline bool FrontierFinder::inmap(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->isInMap(idx);
}

}  // namespace fast_planner