/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkPatientInfoWidget_h
#define QmitkPatientInfoWidget_h

// semantic relations UI module
#include "MitkSemanticRelationsUIExports.h"
#include <ui_QmitkPatientInfoWidgetControls.h>

// qt
#include <QWidget>

// mitk core
#include <mitkDataNode.h>

/*
* @brief The QmitkPatientInfoWidget is a widget that shows some DICOM information about a specific patient (data node).
*
* CURRENTLY NOT USED
*/
class MITKSEMANTICRELATIONSUI_EXPORT QmitkPatientInfoWidget : public QWidget
{
  Q_OBJECT

public:

  QmitkPatientInfoWidget(QWidget* parent = nullptr);
  ~QmitkPatientInfoWidget();

  void SetPatientInfo(const mitk::DataNode* dataNode);

private:

  void QmitkPatientInfoWidget::Init();

  Ui::QmitkPatientInfoWidgetControls m_Controls;

};

#endif
