/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkAutoCropImageFilter_h
#define mitkAutoCropImageFilter_h

#include "MitkAlgorithmsExtExports.h"
#include "mitkCommon.h"
#include "mitkImageTimeSelector.h"
#include "mitkSubImageSelector.h"

#include <itkImage.h>
#include <itkImageRegion.h>

namespace mitk
{
  /**
   *
   * @brief Shrink the image borders to a minimum considering a background color.
   *
   * This filter determines the smallest bounding box of all pixels different
   * from the background, and returns an output image which has been cropped to this size.
   * The box calculated this way is not the smallest possible box, but the box with the
   * smallest sides perpendicular to the world coordinate system.
   *
   * The filter works on 3D and 4D image data. For the 4D case, the smallest box is
   * calculated with side lengths as the maximum of single side lengths from all time steps.
   *
   * 2D images are not supported, and will never be.
   *
   * It is also possible to set the region to be cropped manually using the
   * SetCroppingRegion() method.
   *
   * A margin can be set to enlarge the cropped region with a constant factor in all
   * directions around the smallest possible.
   *
   *
   * @ingroup Process
   *
   * @author Thomas Boettger; revised by Tobias Schwarz and Daniel Stein
   *
   */

  class MITKALGORITHMSEXT_EXPORT AutoCropImageFilter : public SubImageSelector
  {
  public:
    typedef itk::ImageRegion<3> RegionType;

    mitkClassMacro(AutoCropImageFilter, SubImageSelector);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

    itkGetConstMacro(BackgroundValue, float);
    itkSetMacro(BackgroundValue, float);

    itkGetConstMacro(MarginFactor, float);
    itkSetMacro(MarginFactor, float);

    itkGetMacro(CroppingRegion, RegionType);

    // Use this method to manually set a region
    void SetCroppingRegion(RegionType overrideRegion);

    virtual const PixelType GetOutputPixelType();

  protected:
    // default constructor
    AutoCropImageFilter();

    // default destructor
    ~AutoCropImageFilter() override;

    // This method calculates the actual smallest box
    void ComputeNewImageBounds();

    // Crops the image using the itk::RegionOfInterestImageFilter and creates the new output image
    template <typename TPixel, unsigned int VImageDimension>
    void ITKCrop3DImage(itk::Image<TPixel, VImageDimension> *inputItkImage, unsigned int timestep);

    // Here, the output image is initialized by the input and the newly calculated region
    void GenerateOutputInformation() override;

    // Purposely not implemented
    void GenerateInputRequestedRegion() override;

    // Crops the image on all time steps
    void GenerateData() override;

    float m_BackgroundValue;

    RegionType m_CroppingRegion;

    float m_MarginFactor;

    typedef itk::Image<float, 3> ImageType;
    typedef ImageType::Pointer ImagePointer;

    RegionType::SizeType m_RegionSize;
    RegionType::IndexType m_RegionIndex;

    mitk::ImageTimeSelector::Pointer m_TimeSelector;

    mitk::SlicedData::RegionType m_InputRequestedRegion;
    itk::TimeStamp m_TimeOfHeaderInitialization;

    bool m_OverrideCroppingRegion;
  };

} // namespace mitk

#endif
