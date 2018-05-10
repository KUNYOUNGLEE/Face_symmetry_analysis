#include "stdafx.h"

#include <MicroexpressionDetectorFunc.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

using namespace boost::filesystem;

using namespace std;

int max_eyebrow[5];
int max_eye[6];
int max_nose[2];
int max_mouth[8];
int max_outline[8];

int min_eyebrow[5];
int min_eye[6];
int min_nose[2];
int min_mouth[8];
int min_outline[8];

namespace LandmarkDetector
{

	// For subpixel accuracy drawing
	const int draw_shiftbits_micro = 4;
	const int draw_multiplier_micro = 1 << 4;

	//===========================================================================
	// Visualisation functions
	//===========================================================================
	void ProjectDistance(cv::Mat_<double>& dest, const cv::Mat_<double>& mesh, double fx, double fy, double cx, double cy)
	{
		dest = cv::Mat_<double>(mesh.rows, 2, 0.0);

		int num_points = mesh.rows;

		double X, Y, Z;


		cv::Mat_<double>::const_iterator mData = mesh.begin();
		cv::Mat_<double>::iterator projected = dest.begin();

		for (int i = 0; i < num_points; i++)
		{
			// Get the points
			X = *(mData++);
			Y = *(mData++);
			Z = *(mData++);

			double x;
			double y;

			// if depth is 0 the projection is different
			if (Z != 0)
			{
				x = ((X * fx / Z) + cx);
				y = ((Y * fy / Z) + cy);
			}
			else
			{
				x = X;
				y = Y;
			}

			// Project and store in dest matrix
			(*projected++) = x;
			(*projected++) = y;
		}

	}

	// Computing landmarks (to be drawn later possibly)
	vector<cv::Point2d> CalculateLandmarksDistance(const cv::Mat_<double>& shape2D, cv::Mat_<int>& visibilities)
	{
		int n = shape2D.rows / 2;
		vector<cv::Point2d> landmarks;

		for (int i = 0; i < n; ++i)
		{
			if (visibilities.at<int>(i))
			{
				cv::Point2d featurePoint(shape2D.at<double>(i), shape2D.at<double>(i + n));

				landmarks.push_back(featurePoint);
			}
		}
		return landmarks;
	}

	// Computing landmarks (to be drawn later possibly)
	vector<cv::Point2d> CalculateLandmarksDistance(cv::Mat img, const cv::Mat_<double>& shape2D)
	{
		int n;
		vector<cv::Point2d> landmarks;

		if (shape2D.cols == 2)
		{
			n = shape2D.rows;
		}
		else if (shape2D.cols == 1)
		{
			n = shape2D.rows / 2;
		}

		for (int i = 0; i < n; ++i)
		{
			cv::Point2d featurePoint;
			if (shape2D.cols == 1)
			{
				featurePoint = cv::Point2d(shape2D.at<double>(i), shape2D.at<double>(i + n));
			}
			else
			{
				featurePoint = cv::Point2d(shape2D.at<double>(i, 0), shape2D.at<double>(i, 1));
			}
			landmarks.push_back(featurePoint);
		}
		return landmarks;
	}

	// Computing landmarks (to be drawn later possibly)
	vector<cv::Point2d> CalculateLandmarksDistance(CLNF& clnf_model)
	{

		int idx = clnf_model.patch_experts.GetViewIdx(clnf_model.params_global, 0);

		// Because we only draw visible points, need to find which points patch experts consider visible at a certain orientation
		return CalculateLandmarksDistance(clnf_model.detected_landmarks, clnf_model.patch_experts.visibilities[0][idx]);

	}

	int eyebrow_vec[10] = { 21,22,20,23,19,24,18,25,17,26 };
	int eye_vec[12] = { 39,42,38,43,37,44,36,45,41,46,40,47 };
	int nose_vec[4] = { 31,35,32,34 };
	int mouth_vec[16] = { 50,52,49,53,48,54,59,55,58,56,60,64,61,63,67,65 };
	int line_vec[16] = {0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9};

	bool search_part(int arr[], int row, int col, int search_num)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				if (arr[i*col + j] == search_num)
				{
					return true;
				}
			}
		}

		return false;
	}

	void DrawVecs(cv::Mat img,const cv::Mat_<double>& shape2D, cv::Mat_<double>& landmark_3D, const cv::Mat_<int>& visibilities, cv::Point3d vecs[], int point_num)
	{
		cv::Point3d tmp;
		cv::Point origin;

		if (visibilities.at<int>(27) && visibilities.at<int>(30))
		{
			origin.x = cvRound(shape2D.at<double>(27) * (double)draw_multiplier_micro);
			origin.y = cvRound(shape2D.at<double>(27 + point_num) * (double)draw_multiplier_micro);

			for (int i = 0; i < point_num; i++)
			{
				if (visibilities.at<int>(i))
				{
					tmp.x = cvRound(landmark_3D.at<double>(i) * (double)draw_multiplier_micro - landmark_3D.at<double>(27) * (double)draw_multiplier_micro);
					tmp.y = cvRound(landmark_3D.at<double>(i + point_num) * (double)draw_multiplier_micro - landmark_3D.at<double>(27 + point_num) * (double)draw_multiplier_micro);
					tmp.z = cvRound(landmark_3D.at<double>(i + point_num*2) * (double)draw_multiplier_micro - landmark_3D.at<double>(27 + point_num*2) * (double)draw_multiplier_micro);
					vecs[i] = tmp;

					//기준 벡터, 코의 시작점(원점)부터 코 가운데 끝까지의 벡터
					cv::arrowedLine(img, origin, cv::Point(cvRound(shape2D.at<double>(30) * (double)draw_multiplier_micro),
						cvRound(shape2D.at<double>(30 + point_num) * (double)draw_multiplier_micro)), CV_RGB(0, 0, 0), 1, CV_AA, draw_shiftbits_micro, 0.1);

					if (search_part(eyebrow_vec, 5, 2, i))
					{
						cv::arrowedLine(img, origin, cv::Point(cvRound(shape2D.at<double>(i) * (double)draw_multiplier_micro),
							cvRound(shape2D.at<double>(i + point_num) * (double)draw_multiplier_micro)), CV_RGB(0, 255, 0), 0.001, CV_AA, draw_shiftbits_micro, 0.02);
					}
					else if (search_part(eye_vec, 6, 2, i))
					{
						cv::arrowedLine(img, origin, cv::Point(cvRound(shape2D.at<double>(i) * (double)draw_multiplier_micro),
							cvRound(shape2D.at<double>(i + point_num) * (double)draw_multiplier_micro)), CV_RGB(255, 0, 0), 0.001, CV_AA, draw_shiftbits_micro, 0.02);
					}
					else if (search_part(nose_vec, 2, 2, i))
					{
						cv::arrowedLine(img, origin, cv::Point(cvRound(shape2D.at<double>(i) * (double)draw_multiplier_micro),
							cvRound(shape2D.at<double>(i + point_num) * (double)draw_multiplier_micro)), CV_RGB(128, 0, 0), 0.001, CV_AA, draw_shiftbits_micro, 0.02);
					}
					else if (search_part(mouth_vec, 8, 2, i))
					{
						cv::arrowedLine(img, origin, cv::Point(cvRound(shape2D.at<double>(i) * (double)draw_multiplier_micro),
							cvRound(shape2D.at<double>(i + point_num) * (double)draw_multiplier_micro)), CV_RGB(0, 0, 128), 0.001, CV_AA, draw_shiftbits_micro, 0.02);
					}
					else if (search_part(line_vec, 8, 2, i))
					{
						cv::arrowedLine(img, origin, cv::Point(cvRound(shape2D.at<double>(i) * (double)draw_multiplier_micro),
							cvRound(shape2D.at<double>(i + point_num) * (double)draw_multiplier_micro)), CV_RGB(128, 128, 128), 0.1, 4, draw_shiftbits_micro, 0.02);
					}


				}
			}
		}

	}

	void init_min_max()
	{
		for (int i = 0; i < 5; i++)
		{
			max_eyebrow[i] = 0;
			min_eyebrow[i] = 99999;
		}

		for (int i = 0; i < 6; i++)
		{
			max_eye[i] = 0;
			min_eye[i] = 99999;
		}

		for (int i = 0; i < 2; i++)
		{
			max_nose[i] = 0;
			min_nose[i] = 99999;
		}

		for (int i = 0; i < 8; i++)
		{
			max_mouth[i] = 0;
			max_outline[i] = 0;
			min_mouth[i] = 99999;
			min_outline[i] = 99999;
		}
	}

	void calc_min_max(double eyebrow[], double eye[], double nose[], double mouth[], double outline[])
	{
		for (int i = 0; i < 5; i++)
		{
			if (eyebrow[i] > max_eyebrow[i])
				max_eyebrow[i] = eyebrow[i];
			
			if (eyebrow[i] < min_eyebrow[i])
				min_eyebrow[i] = eyebrow[i];
		}
		for (int i = 0; i < 6; i++)
		{
			if (eye[i] > max_eye[i])
				max_eye[i] = eye[i];
			
			if (eye[i] < min_eye[i])
				min_eye[i] = eye[i];
		}
		for (int i = 0; i < 2; i++)
		{
			if (nose[i] > max_nose[i])
				max_nose[i] = nose[i];
			
			if (nose[i] < min_nose[i])
				min_nose[i] = nose[i];
		}
		for (int i = 0; i < 8; i++)
		{
			if (mouth[i] > max_mouth[i])
				max_mouth[i] = mouth[i];
			
			if (mouth[i] < min_mouth[i])
				min_mouth[i] = mouth[i];

			if (outline[i] > max_outline[i])
				max_outline[i] = outline[i];

			if (outline[i] < min_outline[i])
				min_outline[i] = outline[i];
		}
	}

	void calcAsym(int innerdot[], double eyebrow[], double eye[], double nose[], double mouth[], double outline[])
	{
		eyebrow[0] = abs(innerdot[21] - innerdot[22]);
		eyebrow[1] = abs(innerdot[20] - innerdot[23]);
		eyebrow[2] = abs(innerdot[19] - innerdot[24]);
		eyebrow[3] = abs(innerdot[18] - innerdot[25]);
		eyebrow[4] = abs(innerdot[17] - innerdot[26]);

		eye[0] = abs(innerdot[39] - innerdot[42]);
		eye[1] = abs(innerdot[38] - innerdot[43]);
		eye[2] = abs(innerdot[37] - innerdot[44]);
		eye[3] = abs(innerdot[36] - innerdot[45]);
		eye[4] = abs(innerdot[41] - innerdot[46]);
		eye[5] = abs(innerdot[40] - innerdot[47]);

		nose[0] = abs(innerdot[31] - innerdot[35]);
		nose[1] = abs(innerdot[32] - innerdot[34]);

		mouth[0] = abs(innerdot[50] - innerdot[52]);
		mouth[1] = abs(innerdot[49] - innerdot[53]);
		mouth[2] = abs(innerdot[48] - innerdot[54]);
		mouth[3] = abs(innerdot[59] - innerdot[55]);
		mouth[4] = abs(innerdot[58] - innerdot[56]);
		mouth[5] = abs(innerdot[67] - innerdot[65]);
		mouth[6] = abs(innerdot[60] - innerdot[64]);
		mouth[7] = abs(innerdot[61] - innerdot[63]);

		outline[0] = abs(innerdot[0] - innerdot[16]);
		outline[1] = abs(innerdot[1] - innerdot[15]);
		outline[2] = abs(innerdot[2] - innerdot[14]);
		outline[3] = abs(innerdot[3] - innerdot[13]);
		outline[4] = abs(innerdot[4] - innerdot[12]);
		outline[5] = abs(innerdot[5] - innerdot[11]);
		outline[6] = abs(innerdot[6] - innerdot[10]);
		outline[7] = abs(innerdot[7] - innerdot[9]);

		calc_min_max(eyebrow, eye, nose, mouth, outline);
	}

	void drawAsym(cv::Mat graph, double eyebrow[], double eye[], double nose[], double mouth[], double outline[])
	{
		int width = graph.cols;
		int height = graph.rows;

		//eyebrow 0
		cv::rectangle(graph, cv::Point(2, int(height / 2 - eyebrow[0] * 100)), cv::Point(int(width / 13), int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//eyebrow 1
		cv::rectangle(graph, cv::Point(int(width / 13), int(height / 2 - eyebrow[1] * 200)), cv::Point(int(width / 13)*2, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//eyebrow 2
		cv::rectangle(graph, cv::Point(int(width / 13) * 2, int(height / 2 - eyebrow[2] * 200)), cv::Point(int(width / 13) * 3, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//eyebrow 3
		cv::rectangle(graph, cv::Point(int(width / 13) * 3, int(height / 2 - eyebrow[3] * 200)), cv::Point(int(width / 13) * 4, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//eyebrow 4
		cv::rectangle(graph, cv::Point(int(width / 13) * 4, int(height / 2 - eyebrow[4] * 200)), cv::Point(int(width / 13) * 5, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);

		//nose 0
		cv::rectangle(graph, cv::Point(int(width / 13) * 5, int(height / 2 - nose[0] * 200)), cv::Point(int(width / 13) * 6, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//nose 1
		cv::rectangle(graph, cv::Point(int(width / 13) * 6, int(height / 2 - nose[1] * 200)), cv::Point(int(width / 13) * 7, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);

		//eye 0
		cv::rectangle(graph, cv::Point(int(width / 13) * 7, int(height / 2 - eye[0] * 200)), cv::Point(int(width / 13) * 8, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//eye 1
		cv::rectangle(graph, cv::Point(int(width / 13) * 8, int(height / 2 - eye[1] * 200)), cv::Point(int(width / 13) * 9, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//eye 2
		cv::rectangle(graph, cv::Point(int(width / 13) * 9, int(height / 2 - eye[2] * 200)), cv::Point(int(width / 13) * 10, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//eye 3
		cv::rectangle(graph, cv::Point(int(width / 13) * 10, int(height / 2 - eye[3] * 200)), cv::Point(int(width / 13) * 11, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//eye 4
		cv::rectangle(graph, cv::Point(int(width / 13) * 11, int(height / 2 - eye[4] * 200)), cv::Point(int(width / 13) * 12, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		//eye 5
		cv::rectangle(graph, cv::Point(int(width / 13) * 12, int(height / 2 - eye[5] * 200)), cv::Point(int(width / 13) * 13, int(height / 2 )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);

		//mouth 0
		if ((height - mouth[0] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(2, int(height / 2 + 30)), cv::Point(int(width / 16), int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(2, int(height - mouth[0] * 200)), cv::Point(int(width / 16), int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		//mouth 1
		if ((height - mouth[1] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16), int(height / 2 + 30)), cv::Point(int(width / 16) * 2, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16), int(height - mouth[1] * 200)), cv::Point(int(width / 16) * 2, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//mouth 2
		if ((height - mouth[2] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 2, int(height / 2 + 30)), cv::Point(int(width / 16) * 3, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 2, int(height - mouth[2] * 200)), cv::Point(int(width / 16) * 3, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//mouth 3
		if ((height - mouth[3] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 3, int(height / 2 + 30)), cv::Point(int(width / 16) * 4, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 3, int(height - mouth[3] * 200)), cv::Point(int(width / 16) * 4, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//mouth 4
		if ((height - mouth[4] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 4, int(height / 2 + 30)), cv::Point(int(width / 16) * 5, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 4, int(height - mouth[4] * 200)), cv::Point(int(width / 16) * 5, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//mouth 5
		if ((height - mouth[5] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 5, int(height / 2 + 30)), cv::Point(int(width / 16) * 6, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 5, int(height - mouth[5] * 200)), cv::Point(int(width / 16) * 6, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//mouth 6
		if ((height - mouth[6] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 6, int(height / 2 + 30)), cv::Point(int(width / 16) * 7, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 6, int(height - mouth[6] * 200)), cv::Point(int(width / 16) * 7, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//mouth 7
		if ((height - mouth[7] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 7, int(height / 2 + 30)), cv::Point(int(width / 16) * 8, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 7, int(height - mouth[7] * 200)), cv::Point(int(width / 16) * 8, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//outline 0
		if ((height - outline[0] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 8 + 2, int(height / 2 + 30)), cv::Point(int(width / 16) * 9, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 8 + 2, int(height - outline[0] * 200)), cv::Point(int(width / 16) * 9, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//outline 1
		if ((height - outline[1] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 9, int(height / 2 + 30)), cv::Point(int(width / 16) * 10, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 9, int(height - outline[1] * 200)), cv::Point(int(width / 16) * 10, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//outline 2
		if ((height - outline[2] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 10, int(height / 2 + 30)), cv::Point(int(width / 16) * 11, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 10, int(height - outline[2] * 200)), cv::Point(int(width / 16) * 11, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//outline 3
		if ((height - outline[3] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 11, int(height / 2 + 30)), cv::Point(int(width / 16) * 12, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 11, int(height - outline[3] * 200)), cv::Point(int(width / 16) * 12, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//outline 4
		if ((height - outline[4] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 12, int(height / 2 + 30)), cv::Point(int(width / 16) * 13, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 12, int(height - outline[4] * 200)), cv::Point(int(width / 16) * 13, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//outline 5
		if ((height - outline[5] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 13, int(height / 2 + 30)), cv::Point(int(width / 16) * 14, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 13, int(height - outline[5] * 200)), cv::Point(int(width / 16) * 14, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		//outline 6
		if ((height - outline[6] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 14, int(height / 2 + 30)), cv::Point(int(width / 16) * 15, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 14, int(height - outline[6] * 200)), cv::Point(int(width / 16) * 15, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		
		//outline 7
		if ((height - outline[7] * 200) < (height / 2))
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 15, int(height / 2 + 30)), cv::Point(int(width / 16) * 16, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}
		else
		{
			cv::rectangle(graph, cv::Point(int(width / 16) * 15, int(height - outline[7] * 200)), cv::Point(int(width / 16) * 16, int(height )), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		cv::line(graph, cv::Point(1, 1), cv::Point(width + 1, 1), CV_RGB(255, 0, 0), 2);
		cv::line(graph, cv::Point(0, height - 2), cv::Point(width - 2, height - 2), CV_RGB(255, 0, 0), 2);
		cv::line(graph, cv::Point(1, 1), cv::Point(1, height + 1), CV_RGB(255, 0, 0), 2);
		cv::line(graph, cv::Point(width - 2, 0), cv::Point(width - 2, height - 2), CV_RGB(255, 0, 0), 2);

		cv::line(graph, cv::Point(0, (height / 2)), cv::Point(width, (height / 2)), CV_RGB(255, 0, 0), 2);
		cv::line(graph, cv::Point(5 * (width / 13), 0), cv::Point(5 * (width / 13), (height / 2)), CV_RGB(255, 0, 0), 1);
		cv::line(graph, cv::Point(7 * (width / 13), 0), cv::Point(7 * (width / 13), (height / 2)), CV_RGB(255, 0, 0), 1);
		cv::line(graph, cv::Point((width / 2), (height / 2)), cv::Point((width / 2), height), CV_RGB(255, 0, 0), 1);

		char label[255];
		sprintf(label, "Eyebrow");
		cv::putText(graph, label, cv::Point(5, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
		sprintf(label, "Nose");
		cv::putText(graph, label, cv::Point(5 * (width / 13) + 5, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
		sprintf(label, "Eye");
		cv::putText(graph, label, cv::Point(7 * (width / 13) + 5, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
		sprintf(label, "Mouth");
		cv::putText(graph, label, cv::Point(5, (height / 2) + 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
		sprintf(label, "Outline");
		cv::putText(graph, label, cv::Point((width / 2) + 5, (height / 2) + 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
	}

	//영상을 저장할 경로명 문자열을 받아서 상위 디렉토리부터 하위 디렉토리순으로 하나씩 디렉토리를 생성하는 함수
	void MakeDirectory(wchar_t *full_path)
	{
		wchar_t temp[256], *sp;
		wcscpy(temp, full_path);
		//strcpy(temp, full_path); // 경로문자열을 복사
		sp = temp; // 포인터를 문자열 처음으로
		setlocale(LC_ALL, "korean");

		while ((sp = wcschr(sp, '\\'))) { // 디렉토리 구분자를 찾았으면
			if (sp > temp && *(sp - 1) != ':') { // 루트디렉토리가 아니면
				*sp = '\0'; // 잠시 문자열 끝으로 설정
							//CreateDirectory(temp, NULL);
				CreateDirectoryW(temp, NULL);
				// 디렉토리를 만들고 (존재하지 않을 때)
				*sp = '\\'; // 문자열을 원래대로 복귀
			}
			sp++; // 포인터를 다음 문자로 이동
		}
	}

	void CharToWChar(const char* pstrSrc, wchar_t pwstrDest[])
	{
		int nLen = (int)strlen(pstrSrc) + 1;
		mbstowcs(pwstrDest, pstrSrc, nLen);
	}

	std::string TCHARToString(const TCHAR* ptsz)
	{
		int len = wcslen((wchar_t*)ptsz);
		char* psz = new char[2 * len + 1];
		wcstombs(psz, (wchar_t*)ptsz, 2 * len + 1);
		std::string s = psz;
		delete[] psz;
		return s;
	}

	const std::string currentDateTime() {
		time_t     now = time(0);
		struct tm  tstruct;
		char       buf[80];
		localtime_s(&tstruct, &now);
		// Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
		// for more information about date/time format
		strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M", &tstruct);

		return buf;
	}

	double epsilon = 0.00001;

	//내적차이를 정규화하는 함수(min-max 정규화)
	void Norm_Asym(double eyebrow[], double eye[], double nose[], double mouth[], double outline[])
	{
		eyebrow[0] = abs((eyebrow[0] - min_eyebrow[0]) / (max_eyebrow[0] - min_eyebrow[0] + epsilon));
		eyebrow[1] = abs((eyebrow[1] - min_eyebrow[1]) / (max_eyebrow[1] - min_eyebrow[1] + epsilon));
		eyebrow[2] = abs((eyebrow[2] - min_eyebrow[2]) / (max_eyebrow[2] - min_eyebrow[2] + epsilon));
		eyebrow[3] = abs((eyebrow[3] - min_eyebrow[3]) / (max_eyebrow[3] - min_eyebrow[3] + epsilon));
		eyebrow[4] = abs((eyebrow[4] - min_eyebrow[4]) / (max_eyebrow[4] - min_eyebrow[4] + epsilon));

		eye[0] = abs((eye[0] - min_eye[0]) / (max_eye[0] - min_eye[0] + epsilon));
		eye[1] = abs((eye[1] - min_eye[1]) / (max_eye[1] - min_eye[1] + epsilon));
		eye[2] = abs((eye[2] - min_eye[2]) / (max_eye[2] - min_eye[2] + epsilon));
		eye[3] = abs((eye[3] - min_eye[3]) / (max_eye[3] - min_eye[3] + epsilon));
		eye[4] = abs((eye[4] - min_eye[4]) / (max_eye[4] - min_eye[4] + epsilon));
		eye[5] = abs((eye[5] - min_eye[5]) / (max_eye[5] - min_eye[5] + epsilon));

		nose[0] = abs((nose[0] - min_nose[0]) / (max_nose[0] - min_nose[0] + epsilon));
		nose[1] = abs((nose[1] - min_nose[1]) / (max_nose[1] - min_nose[1] + epsilon));

		mouth[0] = abs((mouth[0] - min_mouth[0]) / (max_mouth[0] - min_mouth[0] + epsilon));
		mouth[1] = abs((mouth[1] - min_mouth[1]) / (max_mouth[1] - min_mouth[1] + epsilon));
		mouth[2] = abs((mouth[2] - min_mouth[2]) / (max_mouth[2] - min_mouth[2] + epsilon));
		mouth[3] = abs((mouth[3] - min_mouth[3]) / (max_mouth[3] - min_mouth[3] + epsilon));
		mouth[4] = abs((mouth[4] - min_mouth[4]) / (max_mouth[4] - min_mouth[4] + epsilon));
		mouth[5] = abs((mouth[5] - min_mouth[5]) / (max_mouth[5] - min_mouth[5] + epsilon));
		mouth[6] = abs((mouth[6] - min_mouth[6]) / (max_mouth[6] - min_mouth[6] + epsilon));
		mouth[7] = abs((mouth[7] - min_mouth[7]) / (max_mouth[7] - min_mouth[7] + epsilon));

		outline[0] = abs((outline[0] - min_outline[0]) / (max_outline[0] - min_outline[0] + epsilon));
		outline[1] = abs((outline[1] - min_outline[1]) / (max_outline[1] - min_outline[1] + epsilon));
		outline[2] = abs((outline[2] - min_outline[2]) / (max_outline[2] - min_outline[2] + epsilon));
		outline[3] = abs((outline[3] - min_outline[3]) / (max_outline[3] - min_outline[3] + epsilon));
		outline[4] = abs((outline[4] - min_outline[4]) / (max_outline[4] - min_outline[4] + epsilon));
		outline[5] = abs((outline[5] - min_outline[5]) / (max_outline[5] - min_outline[5] + epsilon));
		outline[6] = abs((outline[6] - min_outline[6]) / (max_outline[6] - min_outline[6] + epsilon));
		outline[7] = abs((outline[7] - min_outline[7]) / (max_outline[7] - min_outline[7] + epsilon));
	}

	// Drawing landmarks on a face image
	void clacvectors(cv::Mat img, cv::Mat graph, const cv::Mat_<double>& shape2D, cv::Mat_<double>& landmark_3D, const cv::Mat_<int>& visibilities)
	{
		int n = landmark_3D.rows / 3;
		std::string outroot;
		std::string outfile;
		TCHAR NPath[200];
		GetCurrentDirectory(200, NPath);

		setlocale(LC_ALL, "korean");
		// By default write to same directory
		outroot = TCHARToString(NPath);

		outroot = outroot + "\\recording\\" + currentDateTime() + "\\";
		outfile = currentDateTime() + "eyebrow_eye_nose" + ".csv";
		wchar_t dir[255];
		CharToWChar(outroot.c_str(), dir);
		MakeDirectory(dir);

		FILE* upperface = NULL;
		FILE* lowerface = NULL;

		std::string save;
		save = outroot + outfile;
		wchar_t exceldir[255];
		CharToWChar(save.c_str(), exceldir);
		upperface = _wfopen(exceldir, L"w+");

		lowerface = _wfopen(exceldir, L"w+");

		// Drawing feature points
		if (n >= 66)
		{
			// A rough heuristic for drawn point size
			int thickness = (int)std::ceil(3.0* ((double)img.cols) / 640.0);
			int thickness_2 = (int)std::ceil(1.0* ((double)img.cols) / 640.0);

			cv::Point3d vecs[68];
			int innerDot[68];
			double eyebrow_Asym[5];
			double eye_Asym[6];
			double nose_Asym[2];
			double mouth_Asym[8];
			double outline_Asym[8];

			for (int i = 0; i < n; i++)
			{
				innerDot[i] = 0;
			}

			//벡터 계산및 이미지로 뿌려줌
			DrawVecs(img, shape2D, landmark_3D, visibilities, vecs, n);

			//내적 계산
			for (int i = 0; i < n; i++)
			{
				innerDot[i] = (int)(vecs[i].x * vecs[30].x) + (int)(vecs[i].y * vecs[30].y) + (int)(vecs[i].z * vecs[30].z);
			}

			calcAsym(innerDot, eyebrow_Asym, eye_Asym, nose_Asym, mouth_Asym, outline_Asym);

			Norm_Asym(eyebrow_Asym, eye_Asym, nose_Asym, mouth_Asym, outline_Asym);

			drawAsym(graph, eyebrow_Asym, eye_Asym, nose_Asym, mouth_Asym, outline_Asym);
		}
	}

	//// Drawing landmarks on a face image
	void DrawDistance(cv::Mat img, const cv::Mat_<double>& shape2D)
	{

		int n;

		if (shape2D.cols == 2)
		{
			n = shape2D.rows;
		}
		else if (shape2D.cols == 1)
		{
			n = shape2D.rows / 2;
		}

		for (int i = 0; i < n; ++i)
		{
			cv::Point featurePoint;
			if (shape2D.cols == 1)
			{
				featurePoint = cv::Point(cvRound(shape2D.at<double>(i) * (double)draw_multiplier_micro), cvRound(shape2D.at<double>(i + n) * (double)draw_multiplier_micro));
			}
			else
			{
				featurePoint = cv::Point(cvRound(shape2D.at<double>(i, 0) * (double)draw_multiplier_micro), cvRound(shape2D.at<double>(i, 1) * (double)draw_multiplier_micro));
			}
			// A rough heuristic for drawn point size
			int thickness = (int)std::ceil(5.0* ((double)img.cols) / 640.0);
			int thickness_2 = (int)std::ceil(1.5* ((double)img.cols) / 640.0);

			cv::circle(img, featurePoint, 1 * draw_multiplier_micro, cv::Scalar(0, 0, 255), thickness, CV_AA, draw_shiftbits_micro);
			cv::circle(img, featurePoint, 1 * draw_multiplier_micro, cv::Scalar(255, 0, 0), thickness_2, CV_AA, draw_shiftbits_micro);

		}

	}

	// Drawing detected landmarks on a face image
	void DrawDistance(cv::Mat img, cv::Mat graph, const CLNF& clnf_model, cv::Mat_<double>& landmark_3D)
	{

		int idx = clnf_model.patch_experts.GetViewIdx(clnf_model.params_global, 0);

		// Because we only draw visible points, need to find which points patch experts consider visible at a certain orientation
		clacvectors(img, graph,clnf_model.detected_landmarks, landmark_3D, clnf_model.patch_experts.visibilities[0][idx]);

		// If the model has hierarchical updates draw those too
		for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
		{
			if (clnf_model.hierarchical_models[i].pdm.NumberOfPoints() != clnf_model.hierarchical_mapping[i].size())
			{
				DrawDistance(img, graph, clnf_model.hierarchical_models[i],landmark_3D );
			}
		}
	}



	void DrawLandmarksDistance(cv::Mat img, vector<cv::Point> landmarks)
	{
		for (cv::Point p : landmarks)
		{

			// A rough heuristic for drawn point size
			int thickness = (int)std::ceil(5.0* ((double)img.cols) / 640.0);
			int thickness_2 = (int)std::ceil(1.5* ((double)img.cols) / 640.0);

			cv::circle(img, p, 1, cv::Scalar(0, 0, 255), thickness, CV_AA);
			cv::circle(img, p, 1, cv::Scalar(255, 0, 0), thickness_2, CV_AA);
		}

	}

}
