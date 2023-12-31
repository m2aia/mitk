/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

// semantic relations plugin
#include "QmitkLesionInfoWidget.h"
#include "QmitkDataNodeAddToSemanticRelationsAction.h"
#include "QmitkFocusOnLesionAction.h"
#include "QmitkSemanticRelationsNodeSelectionDialog.h"

// semantic relations UI module
#include <QmitkLesionTextDialog.h>

// semantic relations module
#include <mitkLesionManager.h>
#include <mitkNodePredicates.h>
#include <mitkSemanticRelationException.h>
#include <mitkSemanticRelationsInference.h>
#include <mitkRelationStorage.h>

// segmentation
#include <mitkLabelSetImage.h>
#include <QmitkNewSegmentationDialog.h>

// qt
#include <QCompleter>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QString>

QmitkLesionInfoWidget::QmitkLesionInfoWidget(mitk::DataStorage* dataStorage,
                                             berry::IWorkbenchPartSite::Pointer workbenchPartSite,
                                             QWidget* parent /*= nullptr*/)
  : QWidget(parent)
  , m_DataStorage(dataStorage)
  , m_WorkbenchPartSite(workbenchPartSite)
  , m_SemanticRelationsDataStorageAccess(std::make_unique<mitk::SemanticRelationsDataStorageAccess>(dataStorage))
  , m_SemanticRelationsIntegration(std::make_unique<mitk::SemanticRelationsIntegration>())
{
  Initialize();
}

void QmitkLesionInfoWidget::Initialize()
{
  m_Controls.setupUi(this);

  m_Controls.lesionTreeView->setAlternatingRowColors(true);
  m_Controls.lesionTreeView->setSelectionMode(QAbstractItemView::SingleSelection);
  m_Controls.lesionTreeView->setSelectionBehavior(QAbstractItemView::SelectRows);
  m_Controls.lesionTreeView->setContextMenuPolicy(Qt::CustomContextMenu);

  m_StorageModel = new QmitkLesionTreeModel(m_Controls.lesionTreeView);
  auto dataStorage = m_DataStorage.Lock();

  if (dataStorage.IsNull())
  {
    return;
  }

  m_StorageModel->SetDataStorage(dataStorage);
  m_Controls.lesionTreeView->setModel(m_StorageModel);

  SetUpConnections();
}

void QmitkLesionInfoWidget::SetUpConnections()
{
  connect(m_StorageModel, &QmitkLesionTreeModel::ModelUpdated, this, &QmitkLesionInfoWidget::OnModelUpdated);
  connect(m_Controls.addLesionPushButton, &QPushButton::clicked, this, &QmitkLesionInfoWidget::OnAddLesionButtonClicked);
  connect(m_Controls.lesionTreeView->selectionModel(), &QItemSelectionModel::currentChanged, this, &QmitkLesionInfoWidget::OnSelectionChanged);
  connect(m_Controls.lesionTreeView, &QTreeView::customContextMenuRequested, this, &QmitkLesionInfoWidget::OnLesionListContextMenuRequested);
}

void QmitkLesionInfoWidget::SetCaseID(const mitk::SemanticTypes::CaseID& caseID)
{
  m_CaseID = caseID;
  m_StorageModel->SetCaseID(caseID);
}

void QmitkLesionInfoWidget::SetDataNodeSelection(const QList<mitk::DataNode::Pointer>& dataNodeSelection)
{
  m_StorageModel->SetDataNodeSelection(dataNodeSelection);
}

//////////////////////////////////////////////////////////////////////////
// Implementation of the QT_SLOTS
//////////////////////////////////////////////////////////////////////////
void QmitkLesionInfoWidget::OnModelUpdated()
{
  m_Controls.lesionTreeView->expandAll();
  int columns = m_Controls.lesionTreeView->model()->columnCount();
  for (int i = 0; i < columns; ++i)
  {
    m_Controls.lesionTreeView->resizeColumnToContents(i);
  }
}

void QmitkLesionInfoWidget::OnAddLesionButtonClicked()
{
  if (m_CaseID.empty())
  {
    QMessageBox msgBox(QMessageBox::Warning,
      "No case ID set.",
      "In order to add a lesion, please specify the current case / patient.");
    msgBox.exec();
    return;
  }

  mitk::SemanticTypes::Lesion newLesion = mitk::GenerateNewLesion();
  try
  {
    m_SemanticRelationsIntegration->AddLesion(m_CaseID, newLesion);
  }
  catch (mitk::SemanticRelationException& e)
  {
    MITK_INFO << "Could not add a new lesion. " << e;
  }
}

void QmitkLesionInfoWidget::OnSelectionChanged(const QModelIndex& current, const QModelIndex& /*previous*/)
{
  // only the UID is needed to identify a representing lesion
  QVariant data = m_StorageModel->data(current, Qt::UserRole);
  if (!data.canConvert<QmitkLesionTreeItem*>())
  {
    return;
  }

  auto lesion = data.value<QmitkLesionTreeItem*>()->GetData().GetLesion();
  if (false == mitk::SemanticRelationsInference::InstanceExists(m_CaseID, lesion))
  {
    // no UID of a existing lesion found; cannot create a lesion
    return;
  }

  // if selected data nodes are set, reset to empty list to
  // hide "selected data nodes presence background highlighting" in the model
  if (!m_StorageModel->GetSelectedDataNodes().isEmpty())
  {
    m_StorageModel->SetDataNodeSelection(QList<mitk::DataNode::Pointer>());
  }

  emit LesionSelectionChanged(lesion);
}

void QmitkLesionInfoWidget::OnLesionListContextMenuRequested(const QPoint& pos)
{
  if (nullptr == m_SemanticRelationsIntegration)
  {
    return;
  }

  if (m_CaseID.empty())
  {
    QMessageBox msgBox(QMessageBox::Warning,
      "No case ID set.",
      "In order to access the context menu entries a case ID has to be set.");
    msgBox.exec();
    return;
  }

  QModelIndex index = m_Controls.lesionTreeView->indexAt(pos);
  if (!index.isValid())
  {
    // no item clicked; cannot retrieve the current lesion
    return;
  }

  QVariant data = m_StorageModel->data(index, Qt::UserRole);
  mitk::SemanticTypes::Lesion selectedLesion;
  if (data.canConvert<QmitkLesionTreeItem*>())
  {
    selectedLesion = data.value<QmitkLesionTreeItem*>()->GetData().GetLesion();
  }
  else
  {
    return;
  }

  QMenu* menu = new QMenu(m_Controls.lesionTreeView);

  QAction* linkToSegmentation = new QAction("Link to segmentation", this);
  linkToSegmentation->setEnabled(true);
  connect(linkToSegmentation, &QAction::triggered, [this, selectedLesion] { OnLinkToSegmentation(selectedLesion); });
  menu->addAction(linkToSegmentation);

  QAction* setLesionName = new QAction("Set lesion name", this);
  setLesionName->setEnabled(true);
  connect(setLesionName, &QAction::triggered, [this, selectedLesion] { OnSetLesionName(selectedLesion); });
  menu->addAction(setLesionName);

  QAction* setLesionClass = new QAction("Set lesion class", this);
  setLesionClass->setEnabled(true);
  connect(setLesionClass, &QAction::triggered, [this, selectedLesion] { OnSetLesionClass(selectedLesion); });
  menu->addAction(setLesionClass);

  QAction* createNewSegmentation = new QAction("Create new lesion", this);
  createNewSegmentation->setEnabled(true);
  connect(createNewSegmentation, &QAction::triggered, [this, selectedLesion] { OnCreateNewSegmentation(selectedLesion); });
  menu->addAction(createNewSegmentation);

  QAction* removeLesion = new QAction("Remove lesion", this);
  removeLesion->setEnabled(true);
  connect(removeLesion, &QAction::triggered, [this, selectedLesion] { OnRemoveLesion(selectedLesion); });
  menu->addAction(removeLesion);

  auto workbenchPartSite = m_WorkbenchPartSite.Lock();

  if (workbenchPartSite.IsNotNull())
  {
    QmitkFocusOnLesionAction* focusOnLesion = new QmitkFocusOnLesionAction(this, workbenchPartSite);
    focusOnLesion->SetDataStorage(m_DataStorage.Lock());
    focusOnLesion->SetSelectedLesion(selectedLesion);
    menu->addAction(focusOnLesion);
  }

  menu->popup(QCursor::pos());
}

void QmitkLesionInfoWidget::OnLinkToSegmentation(mitk::SemanticTypes::Lesion selectedLesion)
{
  auto dataStorage = m_DataStorage.Lock();

  if (dataStorage.IsNull())
  {
    return;
  }

  QmitkSemanticRelationsNodeSelectionDialog* dialog = new QmitkSemanticRelationsNodeSelectionDialog(this, "Select segmentation to link to the selected lesion.", "");
  dialog->setWindowTitle("Select segmentation node");
  dialog->SetDataStorage(dataStorage);
  dialog->SetNodePredicate(mitk::NodePredicates::GetSegmentationPredicate());
  dialog->SetSelectOnlyVisibleNodes(true);
  dialog->SetCaseID(m_CaseID);
  // set the last added segmentation node as pre-selected data node
  const mitk::DataNode* lastSegmentation = m_StorageModel->GetLastSegmentation();
  QList<mitk::DataNode::Pointer> selectedDataNodes;
  if (nullptr != lastSegmentation)
  {
    selectedDataNodes.push_back(const_cast<mitk::DataNode*>(lastSegmentation));
    dialog->SetCurrentSelection(selectedDataNodes);
  }

  int dialogReturnValue = dialog->exec();
  if (QDialog::Rejected == dialogReturnValue)
  {
    return;
  }

  mitk::DataNode::Pointer selectedDataNode = nullptr;
  selectedDataNodes = dialog->GetSelectedNodes();
  if (!selectedDataNodes.isEmpty())
  {
    // only single selection allowed
    selectedDataNode = selectedDataNodes.front();
  }

  if (nullptr == selectedDataNode
    || false == mitk::NodePredicates::GetSegmentationPredicate()->CheckNode(selectedDataNode))
  {
    QMessageBox msgBox(QMessageBox::Warning,
      "No valid segmentation node selected.",
      "In order to link the selected lesion to a segmentation, please specify a valid segmentation node.");
    msgBox.exec();
    return;
  }

  mitk::BaseData* baseData = selectedDataNode->GetData();
  if (nullptr == baseData)
  {
    QMessageBox msgBox(QMessageBox::Warning,
      "No valid base data.",
      "In order to link the selected lesion to a segmentation, please specify a valid segmentation node.");
    msgBox.exec();
    return;
  }

  LinkSegmentationToLesion(selectedDataNode, selectedLesion);
}

void QmitkLesionInfoWidget::OnSetLesionName(mitk::SemanticTypes::Lesion selectedLesion)
{
  // use the lesion information to set the input text for the dialog
  QmitkLesionTextDialog* inputDialog = new QmitkLesionTextDialog(this);
  inputDialog->setWindowTitle("Set lesion name");
  inputDialog->SetLineEditText(selectedLesion.name);

  int dialogReturnValue = inputDialog->exec();
  if (QDialog::Rejected == dialogReturnValue)
  {
    return;
  }

  selectedLesion.name = inputDialog->GetLineEditText().toStdString();
  m_SemanticRelationsIntegration->OverwriteLesion(m_CaseID, selectedLesion);
}

void QmitkLesionInfoWidget::OnSetLesionClass(mitk::SemanticTypes::Lesion selectedLesion)
{
  // use the lesion information to set the input text for the dialog
  QmitkLesionTextDialog* inputDialog = new QmitkLesionTextDialog(this);
  inputDialog->setWindowTitle("Set lesion class");
  inputDialog->SetLineEditText(selectedLesion.lesionClass.classType);

  // prepare the completer for the dialogs input text field
  mitk::LesionClassVector allLesionClasses = mitk::SemanticRelationsInference::GetAllLesionClassesOfCase(m_CaseID);

  QStringList wordList;
  for (const auto& lesionClass : allLesionClasses)
  {
    wordList << QString::fromStdString(lesionClass.classType);
  }
  QCompleter* completer = new QCompleter(wordList, this);
  completer->setCaseSensitivity(Qt::CaseInsensitive);
  inputDialog->GetLineEdit()->setCompleter(completer);

  int dialogReturnValue = inputDialog->exec();
  if (QDialog::Rejected == dialogReturnValue)
  {
    return;
  }

  // retrieve the new input lesion class type and check for an already existing lesion class types
  std::string newLesionClassType = inputDialog->GetLineEditText().toStdString();
  mitk::SemanticTypes::LesionClass existingLesionClass = mitk::FindExistingLesionClass(m_CaseID, newLesionClassType);
  if (existingLesionClass.UID.empty())
  {
    // could not find lesion class information for the new lesion class type
    // create a new lesion class for the selected lesion
    existingLesionClass = mitk::GenerateNewLesionClass(newLesionClassType);
  }

  selectedLesion.lesionClass = existingLesionClass;
  m_SemanticRelationsIntegration->OverwriteLesion(m_CaseID, selectedLesion);
}

void QmitkLesionInfoWidget::OnCreateNewSegmentation(mitk::SemanticTypes::Lesion selectedLesion)
{
  auto dataStorage = m_DataStorage.Lock();

  if (dataStorage.IsNull())
  {
    return;
  }

  QmitkSemanticRelationsNodeSelectionDialog* dialog = new QmitkSemanticRelationsNodeSelectionDialog(this, "Select image to segment lesion on.", "");
  dialog->setWindowTitle("Select image node");
  dialog->SetDataStorage(dataStorage);
  dialog->SetNodePredicate(mitk::NodePredicates::GetImagePredicate());
  dialog->SetSelectOnlyVisibleNodes(true);
  dialog->SetCaseID(m_CaseID);
  dialog->SetLesion(selectedLesion);

  int dialogReturnValue = dialog->exec();
  if (QDialog::Rejected == dialogReturnValue)
  {
    return;
  }

  auto nodes = dialog->GetSelectedNodes();
  mitk::DataNode::Pointer selectedDataNode = nullptr;
  if (!nodes.isEmpty())
  {
    // only single selection allowed
    selectedDataNode = nodes.front();
  }

  if (nullptr == selectedDataNode
    || false == mitk::NodePredicates::GetImagePredicate()->CheckNode(selectedDataNode))
  {
    QMessageBox msgBox(QMessageBox::Warning,
      "No valid image node selected.",
      "In order to create a new segmentation, please specify a valid image node.");
    msgBox.exec();
    return;
  }

  mitk::Image* selectedImage = dynamic_cast<mitk::Image*>(selectedDataNode->GetData());
  if (nullptr == selectedImage)
  {
    QMessageBox msgBox(QMessageBox::Warning,
      "No valid image.",
      "In order to create a new segmentation, please specify a valid image node.");
    msgBox.exec();
    return;
  }

  mitk::LabelSetImage::Pointer segmentation = mitk::LabelSetImage::New();
  try
  {
    segmentation->Initialize(selectedImage);
  }
  catch (mitk::Exception& e)
  {
    std::stringstream exceptionMessage; exceptionMessage << e;
    QMessageBox msgBox(QMessageBox::Warning,
      "Could not initialize segmentation.",
      "The segmentation could not be correctly initialized with the selected image geometry.\n"
      "Reason:\n" + QString::fromStdString(exceptionMessage.str()));
    msgBox.exec();
    return;
  }

  auto segmentationDialog = new QmitkNewSegmentationDialog(this);
  segmentationDialog->setWindowTitle("New lesion segmentation");

  dialogReturnValue = segmentationDialog->exec();
  if (dialogReturnValue == QDialog::Rejected)
  {
    return;
  }

  QString segmentatioName = segmentationDialog->GetName();
  if (segmentatioName.isEmpty())
  {
    segmentatioName = "Unnamed";
  }
  segmentation->GetActiveLabelSet()->AddLabel(segmentatioName.toStdString(), segmentationDialog->GetColor());

  mitk::DataNode::Pointer segmentationNode = mitk::DataNode::New();
  segmentationNode->SetData(segmentation);
  segmentationNode->SetName(segmentatioName.toStdString());
  dataStorage->Add(segmentationNode, selectedDataNode);

  LinkSegmentationToLesion(segmentationNode, selectedLesion);
}

void QmitkLesionInfoWidget::OnRemoveLesion(mitk::SemanticTypes::Lesion selectedLesion)
{
  try
  {
    m_SemanticRelationsIntegration->RemoveLesion(m_CaseID, selectedLesion);
  }
  catch (const mitk::SemanticRelationException& e)
  {
    std::stringstream exceptionMessage; exceptionMessage << e;
    QMessageBox msgBox(QMessageBox::Warning,
      "Could not remove the selected lesion.",
      "The program wasn't able to correctly remove the selected lesion from the semantic relations model.\n"
      "Reason:\n" + QString::fromStdString(exceptionMessage.str()));
    msgBox.exec();
  }
}

void QmitkLesionInfoWidget::LinkSegmentationToLesion(const mitk::DataNode* selectedDataNode, mitk::SemanticTypes::Lesion selectedLesion)
{
  auto dataStorage = m_DataStorage.Lock();

  if (dataStorage.IsNull())
  {
    return;
  }

  // if the segmentation is not contained in the semantic relations, add it
  if (!mitk::SemanticRelationsInference::InstanceExists(selectedDataNode))
  {
    try
    {
      AddToSemanticRelationsAction::Run(dataStorage, selectedDataNode);
    }
    catch (const mitk::SemanticRelationException& e)
    {
      std::stringstream exceptionMessage; exceptionMessage << e;
      QMessageBox msgBox(QMessageBox::Warning,
        "Could not link the selected lesion.",
        "The program wasn't able to correctly link the selected lesion with the selected segmentation.\n"
        "Reason:\n" + QString::fromStdString(exceptionMessage.str() + "\n"));
      msgBox.exec();
    }
  }

  // link the segmentation
  try
  {
    m_SemanticRelationsIntegration->LinkSegmentationToLesion(selectedDataNode, selectedLesion);
  }
  catch (const mitk::SemanticRelationException& e)
  {
    std::stringstream exceptionMessage; exceptionMessage << e;
    QMessageBox msgBox(QMessageBox::Warning,
      "Could not link the selected lesion.",
      "The program wasn't able to correctly link the selected lesion with the selected segmentation.\n"
      "Reason:\n" + QString::fromStdString(exceptionMessage.str()));
    msgBox.exec();
  }
}
