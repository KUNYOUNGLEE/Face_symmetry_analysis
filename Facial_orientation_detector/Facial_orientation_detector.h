
// Facial_orientation_detector.h : PROJECT_NAME ���� ���α׷��� ���� �� ��� �����Դϴ�.
//

#pragma once

#include "LandmarkCoreIncludes.h"
#include "Resource.h"

#include <fstream>
#include <sstream>

#include <iostream>

#include <windows.h>
// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CvvImage.h"

#include <stdio.h>
#include <time.h>

#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#define Button_GetState(hwndCtl)            ((int)(DWORD)SNDMSG((hwndCtl), BM_GETSTATE, 0L, 0L))
#define Button_Enable(hwndCtl, fEnable)         EnableWindow((hwndCtl), (fEnable))

// CFacial_orientation_detectorApp:
// �� Ŭ������ ������ ���ؼ��� Facial_orientation_detector.cpp�� �����Ͻʽÿ�.
//
class CFacial_orientation_detectorApp : public CWinApp
{
public:
	CFacial_orientation_detectorApp();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.
public:

	void printErrorAndAbort(const std::string & error);
	void DisplayImg(HWND dialogwindow, cv::Mat disp_image);
	const std::string currentDateTime();
	vector<string> get_arguments(int argc, char **argv);
	void NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double> >& face_detections);
	bool IsModuleSelected(HWND hwndDlg, const int moduleID);//���̾�α� üũ�ڽ� üũ���� Ȯ���ϴ� �Լ�

	//��ȭ ������ �����ϱ� ���� ������, ������� ����


	DECLARE_MESSAGE_MAP()
};

extern CFacial_orientation_detectorApp theApp;