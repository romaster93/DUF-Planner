#include "gpr.h"

// 기존 4‑parameter 생성자
GPR::GPR(double l, double sigma_f, double sigma_n, double alpha)
  : length_scale_(l), sigma_f_(sigma_f), sigma_n_(sigma_n), alpha_(alpha)
{}

// batch 모드 (기존)
void GPR::addTrainingData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    X_train_ = X;
    y_train_ = y;
    computeKernelInverse();
}

// incremental 모드: 한 점씩 쌓기
void GPR::addTrainingPoint(const Eigen::Vector3d& position, double value) {
    training_points_.push_back(position);
    training_values_.push_back(value);
}

// 쌓인 점 전체로 모델 fitting
void GPR::fitModel() {
    if (training_points_.empty()) return;
    int n = training_points_.size();
    X_train_.resize(n, 3);
    y_train_.resize(n);
    for (int i = 0; i < n; ++i) {
        X_train_.row(i) = training_points_[i].transpose();
        y_train_(i)  = training_values_[i];
    }
    computeKernelInverse();
}

// 커널 행렬 계산 후 역행렬 캐싱
void GPR::computeKernelInverse() {
    int n = X_train_.rows();
    Eigen::MatrixXd K(n, n);
    for (int i = 0; i < n; ++i)
     for (int j = 0; j < n; ++j)
       K(i,j) = kernel(X_train_.row(i), X_train_.row(j));
    K.diagonal().array() += sigma_n_;
    K_inv_ = K.inverse();
}

// 예측(mean)만
double GPR::predict(const Eigen::VectorXd& x_star) {
    int n = X_train_.rows();
    Eigen::VectorXd k_star(n);
    for (int i = 0; i < n; ++i)
      k_star(i) = kernel(X_train_.row(i), x_star);
    return k_star.transpose() * K_inv_ * y_train_;
}

// predictBatch (mean only)
Eigen::VectorXd GPR::predictBatch(const Eigen::MatrixXd& X_star) {
    Eigen::VectorXd result(X_star.rows());
    for (int i = 0; i < X_star.rows(); ++i)
      result(i) = predict(X_star.row(i));
    return result;
}

// predictBatchWithUncertainty (mean + std_dev)
void GPR::predictBatchWithUncertainty(const Eigen::MatrixXd& X_star,
                                      Eigen::VectorXd& mean,
                                      Eigen::VectorXd& std_dev) {
    int m = X_star.rows(), n = X_train_.rows();
    mean.resize(m);
    std_dev.resize(m);
    for (int i = 0; i < m; ++i) {
        // k_star 계산
        Eigen::VectorXd k_star(n);
        for (int j = 0; j < n; ++j)
          k_star(j) = kernel(X_train_.row(j), X_star.row(i));
        double k_ss = kernel(X_star.row(i), X_star.row(i));
        double mu  = k_star.transpose() * K_inv_ * y_train_;
        double var = k_ss - k_star.transpose() * K_inv_ * k_star;
        mean(i)    = mu;
        std_dev(i) = sqrt(std::max(0.0, var));
    }
}

// rational quadratic 커널
double GPR::kernel(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) {
    double d2 = (xi - xj).squaredNorm();
    return 1.0 / (1.0 + alpha_ * d2);
}
