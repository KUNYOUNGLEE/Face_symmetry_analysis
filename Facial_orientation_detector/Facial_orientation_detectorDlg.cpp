
// Facial_orientation_detectorDlg.cpp : ���� ����
//
#include "stdafx.h"
#include "Facial_orientation_detector.h"
#include "Facial_orientation_detectorDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define Button_GetState(hwndCtl)            ((int)(DWORD)SNDMSG((hwndCtl), BM_GETSTATE, 0L, 0L))
#define Button_Enable(hwndCtl, fEnable)         EnableWindow((hwndCtl), (fEnable))

// ���� ���α׷� ������ ���Ǵ� CAboutDlg ��ȭ �����Դϴ�.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// ��ȭ ���� �������Դϴ�.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

														// �����Դϴ�.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()

// CFacial_orientation_detectorDlg ��ȭ ����

CFacial_orientation_detectorDlg::CFacial_orientation_detectorDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_FACIAL_ORIENTATION_DETECTOR_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	playback_mode = false;
	renderer_data.playback_mode = false;
	Runing_thread = 0;
}

void CFacial_orientation_detectorDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CFacial_orientation_detectorDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_START, &CFacial_orientation_detectorDlg::OnBnClickedStart)
	ON_BN_CLICKED(IDC_STOP, &CFacial_orientation_detectorDlg::OnBnClickedStop)
	ON_BN_CLICKED(IDC_BROWSER, &CFacial_orientation_detectorDlg::OnBnClickedBrowser)
	ON_BN_CLICKED(IDC_RESET, &CFacial_orientation_detectorDlg::OnBnClickedReset)
END_MESSAGE_MAP()

// CFacial_orientation_detectorDlg �޽��� ó����

BOOL CFacial_orientation_detectorDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �ý��� �޴��� "����..." �޴� �׸��� �߰��մϴ�.

	// IDM_ABOUTBOX�� �ý��� ��� ������ �־�� �մϴ�.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// �� ��ȭ ������ �������� �����մϴ�.  ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

									// TODO: ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.

	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
}

void CFacial_orientation_detectorDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�.  ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CFacial_orientation_detectorDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ�Դϴ�.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
HCURSOR CFacial_orientation_detectorDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CFacial_orientation_detectorDlg::GetPlaybackFile()
{
	OPENFILENAME filename;
	memset(&filename, 0, sizeof(filename));
	filename.lStructSize = sizeof(filename);
	filename.lpstrFilter = "*.avi\0";
	filename.lpstrFile = fileName;
	fileName[0] = 0;
	filename.nMaxFile = sizeof(fileName) / sizeof(CHAR);
	filename.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_EXPLORER;
	if (!GetOpenFileName(&filename))
		fileName[0] = 0;
	else {
		renderer_data.playback_mode = true; 
		playback_mode = true;
	}
}

void CFacial_orientation_detectorDlg::OnBnClickedStart()
{
	Runing_thread++;
	renderer_data.dialogwindow = AfxGetMainWnd()->m_hWnd;
	renderer_data.Openface_thread_activate = true;
	renderer_data.Reset_activate = false;
	if (playback_mode == true)
	{
		CString str = fileName;
		renderer_data.filename = ((LPCTSTR)str);
		renderer_data.playback_mode = true;
		playback_mode = false;
	}
	else
	{
		CString str = "";
		renderer_data.filename = ((LPCTSTR)str);
		renderer_data.playback_mode = false;
	}

	if (Runing_thread == 1) {

		m_pThread = CreateThread(NULL, NULL, OpenFace, &renderer_data, NULL, dwThreadID);
		if (m_pThread == NULL)
		{
			AfxMessageBox("Error");
		}

		Runing_thread = 0;
	}
}


void CFacial_orientation_detectorDlg::OnBnClickedStop()
{
	renderer_data.Openface_thread_activate = false;
	renderer_data.Reset_activate = false;
	renderer_data.playback_mode = false;
	playback_mode = false;
	Runing_thread = 0;
	fileName[0] = 0;
}


void CFacial_orientation_detectorDlg::OnBnClickedBrowser()
{
	GetPlaybackFile();
}


void CFacial_orientation_detectorDlg::OnBnClickedReset()
{
	renderer_data.Reset_activate = true;
}