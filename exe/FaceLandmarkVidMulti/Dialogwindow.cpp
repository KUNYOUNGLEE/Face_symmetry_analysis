// Dialogwindow.cpp : ���� �����Դϴ�.
//

#include "stdafx.h"
#include "Dialogwindow.h"
#include "afxdialogex.h"


// CDialogwindow ��ȭ �����Դϴ�.

IMPLEMENT_DYNAMIC(CDialogwindow, CDialogEx)

CDialogwindow::CDialogwindow(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_MAINFRAME, pParent)
{

}

CDialogwindow::~CDialogwindow()
{
}

void CDialogwindow::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CDialogwindow, CDialogEx)
END_MESSAGE_MAP()


// CDialogwindow �޽��� ó�����Դϴ�.
