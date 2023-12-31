/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef mitkFloatingImageToUltrasoundRegistrationFilter_h
#define mitkFloatingImageToUltrasoundRegistrationFilter_h

#include <mitkNavigationDataPassThroughFilter.h>
#include <mitkImage.h>

namespace mitk
{
  /**Documentation
  * \brief This filter transforms a given floating image into the ultrasound coordinate system.
  *
  *
  * \ingroup US
  */
  class FloatingImageToUltrasoundRegistrationFilter : public NavigationDataPassThroughFilter
  {
  public:
    mitkClassMacro(FloatingImageToUltrasoundRegistrationFilter, NavigationDataPassThroughFilter);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

    void InitializeTransformationMarkerCSToSensorCS( bool useNdiTracker = false );

    void SetSegmentation(mitk::DataNode::Pointer segmentationNode, mitk::Image::Pointer ctimage);
    void SetSurface(mitk::DataNode::Pointer surfaceNode);
    void SetPointSet(mitk::DataNode::Pointer pointSetNode);
    void SetTransformMarkerCSToFloatingImageCS( mitk::AffineTransform3D::Pointer transform );
    void SetTransformUSimageCSToTrackingCS(mitk::AffineTransform3D::Pointer transform);

  protected:
    FloatingImageToUltrasoundRegistrationFilter();
    ~FloatingImageToUltrasoundRegistrationFilter() override;

    /**Documentation
    * \brief filter execute method
    *
    *
    */
    void GenerateData() override;

    mitk::AffineTransform3D::Pointer GetInverseTransform(mitk::AffineTransform3D::Pointer transform);

  private:
    mitk::DataNode::Pointer m_Segmentation;
    mitk::DataNode::Pointer m_Surface;
    mitk::DataNode::Pointer m_PointSet;
    mitk::Image::Pointer m_CTimage;
    mitk::AffineTransform3D::Pointer m_TransformMarkerCSToSensorCS;
    mitk::AffineTransform3D::Pointer m_TransformMarkerCSToFloatingImageCS;
    mitk::AffineTransform3D::Pointer m_TransformUSimageCSToTrackingCS;
    mitk::AffineTransform3D::Pointer m_TransformCTimageIndexToWorld;
    mitk::BaseGeometry::Pointer m_SurfaceGeometry;
    bool m_TrackedUltrasoundActive;

  };
} // namespace mitk
#endif
