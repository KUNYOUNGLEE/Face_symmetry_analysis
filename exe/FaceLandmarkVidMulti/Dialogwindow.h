#pragma once


// CDialogwindow ��ȭ �����Դϴ�.

class CDialogwindow : public CDialogEx
{
	DECLARE_DYNAMIC(CDialogwindow)

public:
	CDialogwindow(CWnd* pParent = NULL);   // ǥ�� �������Դϴ�.
	virtual ~CDialogwindow();

// ��ȭ ���� �������Դϴ�.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MAINFRAME };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

	DECLARE_MESSAGE_MAP()
};
