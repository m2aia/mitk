/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <mitkMultiLabelSurfaceNetsExtractor.h>

#include <mitkBaseGeometry.h>

#include <vtkImageData.h>
#include <vtkImageWrapPad.h>
#include <vtkMatrix4x4.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkSurfaceNets3D.h>

namespace
{
  void PrepareInput(vtkImageData* groupImage)
  {
    // The vtkImageData wraps an MITK-owned buffer via SetVoidArray; in-place edits do
    // not bump the wrapper's MTime. Force VTK to treat the data as modified so the
    // pipeline re-executes against the current contents.
    if (auto* scalars = groupImage->GetPointData()->GetScalars())
      scalars->Modified();

    groupImage->Modified();
  }

  /** Returns a two-samples-thick copy of a group image that is flat along one or more axes,
      or nullptr if the image is a volume already.

      vtkSurfaceNets3D rejects anything that is not a volume ("Expecting 3D data (volume).")
      and returns nothing for it, which leaves single-slice segmentations - the common case
      for 2D imaging mass spectrometry - invisible in the 3D window. Duplicating the slice
      along its flat axis gives the algorithm the second sample it needs; the result is a
      slab that shows the labels as a plate in 3D.

      The duplicate must not double the thickness or push the slab off the slice, so each
      duplicated axis gets half the original spacing and an origin that centers the two
      samples on the original one. The slab then occupies exactly the voxel the slice
      really has. */
  vtkSmartPointer<vtkImageData> ThickenFlatAxes(vtkImageData* groupImage)
  {
    int extent[6];
    groupImage->GetExtent(extent);

    int paddedExtent[6];
    bool isFlat[3];
    bool anyFlat = false;
    for (int axis = 0; axis < 3; ++axis)
    {
      paddedExtent[2 * axis] = extent[2 * axis];
      paddedExtent[2 * axis + 1] = extent[2 * axis + 1];

      isFlat[axis] = extent[2 * axis] >= extent[2 * axis + 1];
      if (isFlat[axis])
      {
        paddedExtent[2 * axis + 1] = extent[2 * axis] + 1;
        anyFlat = true;
      }
    }

    if (!anyFlat)
      return nullptr;

    // Wrapping repeats the only sample the flat axis has, so the added slice is a copy of it.
    auto pad = vtkSmartPointer<vtkImageWrapPad>::New();
    pad->SetInputData(groupImage);
    pad->SetOutputWholeExtent(paddedExtent);
    pad->Update();

    auto thickened = vtkSmartPointer<vtkImageData>::New();
    thickened->ShallowCopy(pad->GetOutput());

    double spacing[3];
    double origin[3];
    groupImage->GetSpacing(spacing);
    groupImage->GetOrigin(origin);
    for (int axis = 0; axis < 3; ++axis)
    {
      if (!isFlat[axis])
        continue;

      // Keep the pair of samples centered on the position of the original one:
      // point coordinates are origin + index * spacing, so with half the spacing the
      // origin has to move to put the midpoint of the two indices back where the
      // single index was.
      const int firstIndex = extent[2 * axis];
      const double center = origin[axis] + firstIndex * spacing[axis];
      spacing[axis] *= 0.5;
      origin[axis] = center - (firstIndex + 0.5) * spacing[axis];
    }
    thickened->SetSpacing(spacing);
    thickened->SetOrigin(origin);

    return thickened;
  }
}

mitk::MultiLabelSurfaceNetsExtractor::MultiLabelSurfaceNetsExtractor()
  : m_SurfaceNets(vtkSmartPointer<vtkSurfaceNets3D>::New()),
    m_NormalsFilter(vtkSmartPointer<vtkPolyDataNormals>::New()),
    m_Smoothing(true)
{
  m_SurfaceNets->SetOutputMeshTypeToTriangles();

  m_NormalsFilter->SetInputConnection(m_SurfaceNets->GetOutputPort());
  m_NormalsFilter->SplittingOn();
  m_NormalsFilter->ConsistencyOn();
  m_NormalsFilter->AutoOrientNormalsOn();
  m_NormalsFilter->ComputePointNormalsOn();
  m_NormalsFilter->ComputeCellNormalsOff();
  m_NormalsFilter->SetFeatureAngle(30.0);
}

mitk::MultiLabelSurfaceNetsExtractor::~MultiLabelSurfaceNetsExtractor() = default;

void mitk::MultiLabelSurfaceNetsExtractor::SetSmoothing(bool smoothing)
{
  m_Smoothing = smoothing;
}

bool mitk::MultiLabelSurfaceNetsExtractor::GetSmoothing() const
{
  return m_Smoothing;
}

void mitk::MultiLabelSurfaceNetsExtractor::ConfigureLabels(const std::vector<LabelValueType>& labelValues)
{
  m_SurfaceNets->SetSmoothing(m_Smoothing);
  m_SurfaceNets->SetNumberOfLabels(static_cast<int>(labelValues.size()));
  for (size_t i = 0; i < labelValues.size(); ++i)
    m_SurfaceNets->SetLabel(static_cast<int>(i), static_cast<double>(labelValues[i]));
}

vtkSmartPointer<vtkPolyData> mitk::MultiLabelSurfaceNetsExtractor::Extract(
  vtkImageData* groupImage,
  const std::vector<LabelValueType>& labelValues)
{
  auto output = vtkSmartPointer<vtkPolyData>::New();

  if (groupImage == nullptr || labelValues.empty())
  {
    return output;
  }

  PrepareInput(groupImage);

  auto thickened = ThickenFlatAxes(groupImage);

  m_SurfaceNets->SetInputData(thickened != nullptr ? thickened.Get() : groupImage);
  this->ConfigureLabels(labelValues);
  m_SurfaceNets->InitializeSelectedLabelsList();
  m_SurfaceNets->SetOutputStyleToDefault();
  // TODO(VTK 9.5.2): force re-extraction. vtkSurfaceNets3D::RequestData reuses its
  // cache based on Superclass::GetMTime(), which SetLabel/SetInputData do not always
  // bump; the cached path then leaves newScalars null and TransformMeshType crashes.
  // Drop this Modified() when VTK is upgraded past 9.5.2.
  m_SurfaceNets->Modified();

  m_NormalsFilter->Update();
  output->DeepCopy(m_NormalsFilter->GetOutput());

  return output;
}

std::map<mitk::MultiLabelSurfaceNetsExtractor::LabelValueType, vtkSmartPointer<vtkPolyData>>
mitk::MultiLabelSurfaceNetsExtractor::ExtractPerLabel(
  vtkImageData* groupImage,
  const std::vector<LabelValueType>& labelValues)
{
  std::map<LabelValueType, vtkSmartPointer<vtkPolyData>> results;
  if (groupImage == nullptr || labelValues.empty())
  {
    return results;
  }

  PrepareInput(groupImage);

  auto thickened = ThickenFlatAxes(groupImage);

  m_SurfaceNets->SetInputData(thickened != nullptr ? thickened.Get() : groupImage);
  this->ConfigureLabels(labelValues);
  m_SurfaceNets->SetOutputStyleToSelected();

  for (auto label : labelValues)
  {
    m_SurfaceNets->InitializeSelectedLabelsList();
    m_SurfaceNets->AddSelectedLabel(static_cast<double>(label));
    m_SurfaceNets->Modified(); // VTK 9.5.2 cache bust, see Extract().

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    m_NormalsFilter->Update();
    polydata->DeepCopy(m_NormalsFilter->GetOutput());

    if (polydata->GetNumberOfCells() > 0)
      results[label] = polydata;
  }

  return results;
}

vtkSmartPointer<vtkMatrix4x4> mitk::MultiLabelSurfaceNetsExtractor::GetImageToWorldMatrix(const BaseGeometry* geometry)
{
  auto matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  matrix->Identity();
  if (geometry == nullptr)
    return matrix;

  // GetVtkMatrix() returns the index-to-world matrix where the rotation columns are
  // pre-scaled by spacing. vtkSurfaceNets3D consumes a vtkImageData whose spacing is
  // already applied (origin (0,0,0), identity direction), so the polydata it emits is
  // in mm-scaled image-local coordinates. Strip the spacing here to leave a pure
  // [direction | origin] transform that maps those coordinates into world space.
  matrix->DeepCopy(geometry->GetVtkMatrix());
  const auto spacing = geometry->GetSpacing();
  for (int i = 0; i < 3; ++i)
  {
    matrix->SetElement(i, 0, matrix->GetElement(i, 0) / spacing[0]);
    matrix->SetElement(i, 1, matrix->GetElement(i, 1) / spacing[1]);
    matrix->SetElement(i, 2, matrix->GetElement(i, 2) / spacing[2]);
  }
  return matrix;
}
