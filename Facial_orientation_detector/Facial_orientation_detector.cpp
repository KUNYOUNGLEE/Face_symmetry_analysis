
// Facial_orientation_detector.cpp : ���� ���α׷��� ���� Ŭ���� ������ �����մϴ�.
//
#include "stdafx.h"
#include "Facial_orientation_detector.h"
#include "Facial_orientation_detectorDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

// CFacial_orientation_detectorApp

BEGIN_MESSAGE_MAP(CFacial_orientation_detectorApp, CWinApp)
	ON_COMMAND(ID_HELP, &CWinApp::OnHelp)
END_MESSAGE_MAP()

// CFacial_orientation_detectorApp ����

CFacial_orientation_detectorApp::CFacial_orientation_detectorApp()
{
	// �ٽ� ���� ������ ����
	m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_RESTART;

	// InitInstance�� ��� �߿��� �ʱ�ȭ �۾��� ��ġ�մϴ�.
}


// ������ CFacial_orientation_detectorApp ��ü�Դϴ�.

CFacial_orientation_detectorApp theApp;


// CFacial_orientation_detectorApp �ʱ�ȭ

BOOL CFacial_orientation_detectorApp::InitInstance()
{
	// ���� ���α׷� �Ŵ��佺Ʈ�� ComCtl32.dll ���� 6 �̻��� ����Ͽ� ���־� ��Ÿ����
	// ����ϵ��� �����ϴ� ���, Windows XP �󿡼� �ݵ�� InitCommonControlsEx()�� �ʿ��մϴ�.
	// InitCommonControlsEx()�� ������� ������ â�� ���� �� �����ϴ�.
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// ���� ���α׷����� ����� ��� ���� ��Ʈ�� Ŭ������ �����ϵ���
	// �� �׸��� �����Ͻʽÿ�.
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinApp::InitInstance();


	AfxEnableControlContainer();

	// ��ȭ ���ڿ� �� Ʈ�� �� �Ǵ�
	// �� ��� �� ��Ʈ���� ���ԵǾ� �ִ� ��� �� �����ڸ� ����ϴ�.
	CShellManager *pShellManager = new CShellManager;

	// MFC ��Ʈ���� �׸��� ����ϱ� ���� "Windows ����" ���־� ������ Ȱ��ȭ
	CMFCVisualManager::SetDefaultManager(RUNTIME_CLASS(CMFCVisualManagerWindows));

	// ǥ�� �ʱ�ȭ
	// �̵� ����� ������� �ʰ� ���� ���� ������ ũ�⸦ ���̷���
	// �Ʒ����� �ʿ� ���� Ư�� �ʱ�ȭ
	// ��ƾ�� �����ؾ� �մϴ�.
	// �ش� ������ ����� ������Ʈ�� Ű�� �����Ͻʽÿ�.
	// TODO: �� ���ڿ��� ȸ�� �Ǵ� ������ �̸��� ����
	// ������ �������� �����ؾ� �մϴ�.
	SetRegistryKey(_T("���� ���� ���α׷� �����翡�� ������ ���� ���α׷�"));
	CFacial_orientation_detectorDlg dlg;

	m_pMainWnd = &dlg;
	INT_PTR nResponse = dlg.DoModal();

	if (nResponse == -1)
	{
		TRACE(traceAppMsg, 0, "���: ��ȭ ���ڸ� ������ �������Ƿ� ���� ���α׷��� ����ġ �ʰ� ����˴ϴ�.\n");
		TRACE(traceAppMsg, 0, "���: ��ȭ ���ڿ��� MFC ��Ʈ���� ����ϴ� ��� #define _AFX_NO_MFC_CONTROLS_IN_DIALOGS�� ������ �� �����ϴ�.\n");
	}

	//// ������ ���� �� �����ڸ� �����մϴ�.
	//if (pShellManager != NULL)
	//{
	//	delete pShellManager;
	//}

#ifndef _AFXDLL
	ControlBarCleanUp();
#endif

	// ��ȭ ���ڰ� �������Ƿ� ���� ���α׷��� �޽��� ������ �������� �ʰ�  ���� ���α׷��� ���� �� �ֵ��� FALSE��
	// ��ȯ�մϴ�.
	return FALSE;
}

void CFacial_orientation_detectorApp::printErrorAndAbort(const std::string & error)
{
	abort();
}

void CFacial_orientation_detectorApp::DisplayImg(HWND dialogwindow, cv::Mat disp_image)
{
	IplImage* m_pImage = NULL;
	CvvImage m_cImage;

	if (!disp_image.empty())
	{
		m_pImage = &IplImage(disp_image);//cv::Mat -> IplImage
		m_cImage.CopyOf(m_pImage);//IplImage -> cvvImage (Opencv�� cv::Mat�� �پ��� �÷����� ���������� picture control DC�� ����ϱ� ���ؼ� ��ȯ�� �ʿ���)

		HWND panel = GetDlgItem(dialogwindow, IDC_PANEL);
		RECT rc;
		GetClientRect(panel, &rc);//pitcture control �簢���� ����
		HDC dc = GetDC(panel);
		m_cImage.DrawToHDC(dc, &rc);//���
		ReleaseDC(dialogwindow, dc);//dc����
	}
}

const std::string CFacial_orientation_detectorApp::currentDateTime()
{
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	localtime_s(&tstruct, &now);
	// Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
	// for more information about date/time format
	strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M", &tstruct);

	return buf;
}

vector<string> CFacial_orientation_detectorApp::get_arguments(int argc, char ** argv)
{
	vector<string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

void CFacial_orientation_detectorApp::NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double>>& face_detections)
{
	// Go over the model and eliminate detections that are not informative (there already is a tracker there)
	for (size_t model = 0; model < clnf_models.size(); ++model)
	{

		// See if the detections intersect
		cv::Rect_<double> model_rect = clnf_models[model].GetBoundingBox();

		for (int detection = face_detections.size() - 1; detection >= 0; --detection)
		{
			double intersection_area = (model_rect & face_detections[detection]).area();
			double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

			// If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
			if (intersection_area / union_area > 0.5)
			{
				face_detections.erase(face_detections.begin() + detection);
			}
		}
	}
}

bool CFacial_orientation_detectorApp::IsModuleSelected(HWND hwndDlg, const int moduleID)
{
	return (Button_GetState(GetDlgItem(hwndDlg, moduleID)) & BST_CHECKED) != 0;
}



