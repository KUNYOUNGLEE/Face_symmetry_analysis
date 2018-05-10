#pragma once

// OpenCV includes
#include <opencv2/core/core.hpp>

#include "LandmarkDetectorModel.h"

using namespace std;

namespace LandmarkDetector
{
	//===========================================================================
	// Visualisation functions
	//===========================================================================
	void ProjectDistance(cv::Mat_<double>& dest, const cv::Mat_<double>& mesh, double fx, double fy, double cx, double cy);
	vector<cv::Point2d> CalculateLandmarksDistance(const cv::Mat_<double>& shape2D, cv::Mat_<int>& visibilities);
	vector<cv::Point2d> CalculateLandmarksDistance(CLNF& clnf_model);
	void DrawLandmarksDistance(cv::Mat img, vector<cv::Point> landmarks);

	void clacvectors(cv::Mat img, cv::Mat graph, const cv::Mat_<double>& shape2D ,cv::Mat_<double>& landmark_3D, const cv::Mat_<int>& visibilities);
	void DrawDistance(cv::Mat img, const cv::Mat_<double>& shape2D);
	void DrawDistance(cv::Mat img, cv::Mat graph, const CLNF& clnf_model, cv::Mat_<double>& landmark_3D);

	void init_min_max();
	void calc_min_max(int eyebrow[], int eye[], int nose[], int mouth[], int outline[]);
}