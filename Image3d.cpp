#include "Image3d.h"


void Image3d::normalize_image_points(cv::Mat &centers, int num_points, cv::Mat &camera_intrinsics)
{
    // Intrinsic camera parameters
    float fc1 = camera_intrinsics.at<float>(0, 0);
    float fc2 = camera_intrinsics.at<float>(0, 1);
    float cc1 = camera_intrinsics.at<float>(0, 2);
    float cc2 = camera_intrinsics.at<float>(0, 3);
    float alpha_c = camera_intrinsics.at<float>(0, 4);
    float k1 = camera_intrinsics.at<float>(0, 5);
    float k2 = camera_intrinsics.at<float>(0, 6);
    float k3 = camera_intrinsics.at<float>(0, 7);
    float p1 = camera_intrinsics.at<float>(0, 8);
    float p2 = camera_intrinsics.at<float>(0, 9);


    for(int i = 0; i < num_of_points; ++i) {
        // Subtract principal point and divide by the focal length
        float x_distort = (centers.at<float>(i, 0) - cc1) / fc1;
        float y_distort = (centers.at<float>(i, 1) - cc2) / fc2;

        // Correct lens distortion (initial guess is given by the distorted point)
        float x_normalized = x_distort;
        float y_normalized = y_distort;

        for(int k = 0; k < 20; ++k) {
            float r_2 = x_normalized * x_normalized + y_normalized * y_normalized; // Squared L2-Norm
            float k_radial = 1 + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2;
            float delta_x = 2 * p1 * x_normalized * y_normalized + p2 * (r_2 + 2 * x_normalized * x_normalized);
            float delta_y = p1 * (r_2 + 2 * y_normalized * y_normalized) + 2 * p2 * x_normalized * y_normalized;
            x_normalized = (x_distort - delta_x) / k_radial;
            y_normalized = (y_distort - delta_y) / k_radial;
        }

        // Normalized image points
        centers.at<float>(i, 0) = x_normalized;
        centers.at<float>(i, 1) = y_normalized;
    }
}


void Image3d::triangulate(cv::Mat &P1, cv::Mat &P2, cv::Mat &center1, cv::Mat &center2, cv::Mat &center_spatial) {
    // Establish the 4 x 4 matrix A such that A * x = 0
    cv::Mat A_mat = cv::Mat(4, 4, CV_32FC1);
    cv::Mat center_spatial_hmg;

    A_mat.at<float>(0, 0) = P1.at<float>(2, 0) * center1.at<float>(0, 0) - P1.at<float>(0, 0);
    A_mat.at<float>(0, 1) = P1.at<float>(2, 1) * center1.at<float>(0, 0) - P1.at<float>(0, 1);
    A_mat.at<float>(0, 2) = P1.at<float>(2, 2) * center1.at<float>(0, 0) - P1.at<float>(0, 2);
    A_mat.at<float>(0, 3) = P1.at<float>(2, 3) * center1.at<float>(0, 0) - P1.at<float>(0, 3);

    A_mat.at<float>(1, 0) = P1.at<float>(2, 0) * center1.at<float>(0, 1) - P1.at<float>(1, 0);
    A_mat.at<float>(1, 1) = P1.at<float>(2, 1) * center1.at<float>(0, 1) - P1.at<float>(1, 1);
    A_mat.at<float>(1, 2) = P1.at<float>(2, 2) * center1.at<float>(0, 1) - P1.at<float>(1, 2);
    A_mat.at<float>(1, 3) = P1.at<float>(2, 3) * center1.at<float>(0, 1) - P1.at<float>(1, 3);

    A_mat.at<float>(2, 0) = P2.at<float>(2, 0) * center2.at<float>(0, 0) - P2.at<float>(0, 0);
    A_mat.at<float>(2, 1) = P2.at<float>(2, 1) * center2.at<float>(0, 0) - P2.at<float>(0, 1);
    A_mat.at<float>(2, 2) = P2.at<float>(2, 2) * center2.at<float>(0, 0) - P2.at<float>(0, 2);
    A_mat.at<float>(2, 3) = P2.at<float>(2, 3) * center2.at<float>(0, 0) - P2.at<float>(0, 3);

    A_mat.at<float>(3, 0) = P2.at<float>(2, 0) * center2.at<float>(0, 1) - P2.at<float>(1, 0);
    A_mat.at<float>(3, 1) = P2.at<float>(2, 1) * center2.at<float>(0, 1) - P2.at<float>(1, 1);
    A_mat.at<float>(3, 2) = P2.at<float>(2, 2) * center2.at<float>(0, 1) - P2.at<float>(1, 2);
    A_mat.at<float>(3, 3) = P2.at<float>(2, 3) * center2.at<float>(0, 1) - P2.at<float>(1, 3);


    // The sought solution is the eigenvector that corresponds to the smallest singular value
    cv::SVD svd(A_mat);
    center_spatial_hmg = svd.vt.row(svd.vt.rows - 1);


    // Convert homogenous point to euclidean point
    center_spatial.at<float>(0, 0) = center_spatial_hmg.at<float>(0, 0) / center_spatial_hmg.at<float>(0, 3);
    center_spatial.at<float>(0, 1) = center_spatial_hmg.at<float>(0, 1) / center_spatial_hmg.at<float>(0, 3);
    center_spatial.at<float>(0, 2) = center_spatial_hmg.at<float>(0, 2) / center_spatial_hmg.at<float>(0, 3);
}



