#ifndef GPR_NODE_H
#define GPR_NODE_H

#include <vector>
#include <Eigen/Dense>

class GPR {
public:
    /**
     * Constructor with hyperparameters.
     * @param l       Length scale for kernel.
     * @param sigma_f Signal variance.
     * @param sigma_n Noise variance.
     * @param alpha   Kernel parameter for rational quadratic.
     */
    GPR(double l = 1.0,
        double sigma_f = 1.0,
        double sigma_n = 1e-3,
        double alpha = 1.0);

    /**
     * Batch interface: add entire training dataset.
     * @param X Matrix of size [N x 3] with input positions.
     * @param y Vector of size N with observed values.
     */
    void addTrainingData(const Eigen::MatrixXd& X,
                         const Eigen::VectorXd& y);

    /**
     * Incremental interface: add single training point.
     * @param position 3D position vector.
     * @param value    Observed value at position.
     */
    void addTrainingPoint(const Eigen::Vector3d& position,
                          double value);

    /**
     * Fit the GPR model using accumulated training points.
     * Call this before prediction when using incremental data.
     */
    void fitModel();

    /**
     * Predict output at a single query point.
     * @param x_star Query input vector of size 3.
     * @return Predicted mean value.
     */
    double predict(const Eigen::VectorXd& x_star);

    /**
     * Predict outputs (mean only) for a batch of query points.
     * @param X_star Matrix [M x 3] of query points.
     * @return Vector of size M with predicted means.
     */
    Eigen::VectorXd predictBatch(const Eigen::MatrixXd& X_star);

    /**
     * Predict outputs with uncertainty for a batch of query points.
     * @param X_star Matrix [M x 3] of query points.
     * @param mean   Output vector [M] for predicted means.
     * @param std_dev Output vector [M] for predicted standard deviations.
     */
    void predictBatchWithUncertainty(const Eigen::MatrixXd& X_star,
                                     Eigen::VectorXd& mean,
                                     Eigen::VectorXd& std_dev);

private:
    /**
     * Compute and cache inverse of the kernel matrix.
     */
    void computeKernelInverse();

    /**
     * Kernel function (rational quadratic style).
     */
    double kernel(const Eigen::VectorXd& xi,
                  const Eigen::VectorXd& xj);

    // Batch training data
    Eigen::MatrixXd X_train_;
    Eigen::VectorXd y_train_;
    Eigen::MatrixXd K_inv_;

    // Incremental training storage
    std::vector<Eigen::Vector3d> training_points_;
    std::vector<double>           training_values_;

    // Hyperparameters
    double length_scale_;
    double sigma_f_;
    double sigma_n_;
    double alpha_;
};

#endif // GPR_NODE_H
