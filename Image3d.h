#ifndef IMAGE3D_H
#define IMAGE3D_H


#include <cv.h>
#include <opencv2/opencv.hpp>


/*! Utility class for 3D image processing */
class Image3d {
	public:
	    /*! Normalize image points (lens distortion correction)
			\param centers Detected image points
			\param num_points Number of points included in centers
			\param camera_intrinsics Intrinsic camera parameters
	    */
		static void normalize_image_points(cv::Mat &centers, int num_points, cv::Mat &camera_intrinsics);


	    /*! Triangulate a spatial point (Linear-Eigen method)
			\param P1 Projection matrix of the left camera
			\param P2 Projection matrix of the right camera
			\param center1 Image point of the left camera
			\param center2 Corresponding image point of the right camera
			\param center_spatial Triangulated point
	    */
		static void triangulate(cv::Mat &P1, cv::Mat &P2, cv::Mat &center1, cv::Mat &center2, cv::Mat &center_spatial);
};


#endif
