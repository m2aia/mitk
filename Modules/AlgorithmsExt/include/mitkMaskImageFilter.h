/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkMaskImageFilter_h
#define mitkMaskImageFilter_h

#include "MitkAlgorithmsExtExports.h"
#include "mitkCommon.h"
#include "mitkImageTimeSelector.h"
#include "mitkImageToImageFilter.h"

#include "itkImage.h"

namespace mitk
{
  //##Documentation
  //## @brief
  //## @ingroup Process
  class MITKALGORITHMSEXT_EXPORT MaskImageFilter : public ImageToImageFilter
  {
  public:
    mitkClassMacro(MaskImageFilter, ImageToImageFilter);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

      void SetMask(const mitk::Image *mask);
    const mitk::Image *GetMask() const;

    /**
    * get/set the min Value of the original image in the masked area
    **/
    itkGetMacro(MinValue, mitk::ScalarType);
    itkSetMacro(MinValue, mitk::ScalarType);

    /**
    * get/set the max Value of the original image in the masked area
    **/
    itkGetMacro(MaxValue, mitk::ScalarType);
    itkSetMacro(MaxValue, mitk::ScalarType);

    /**
     * This value is used as outside value. This only works
     * if OverrideOutsideValue is set to true. Default is 0.
     **/
    itkSetMacro(OutsideValue, mitk::ScalarType);

    /**
     * This value is used as outside value. This only works
     * if OverrideOutsideValue is set to true. Default is 0.
     */
    itkGetMacro(OutsideValue, mitk::ScalarType);

    /**
     * If OverrideOutsideValue is set to false, this minimum
     * of the pixel type of the output image is taken as outside
     * value. If set to true, the value set via SetOutsideValue is
     * used as background.
     */
    itkSetMacro(OverrideOutsideValue, bool);
    /**
     * If OverrideOutsideValue is set to false, this minimum
     * of the pixel type of the output image is taken as outside
     * value. If set to true, the value set via SetOutsideValue is
     * used as background.
     */
    itkGetMacro(OverrideOutsideValue, bool);
    itkBooleanMacro(OverrideOutsideValue);

  protected:
    MaskImageFilter();

    ~MaskImageFilter() override;

    void GenerateInputRequestedRegion() override;

    void GenerateOutputInformation() override;

    void GenerateData() override;

    template <typename TPixel, unsigned int VImageDimension>
    void InternalComputeMask(itk::Image<TPixel, VImageDimension> *itkImage);

    mitk::Image::Pointer m_Mask;
    mitk::ImageTimeSelector::Pointer m_InputTimeSelector;
    mitk::ImageTimeSelector::Pointer m_MaskTimeSelector;
    mitk::ImageTimeSelector::Pointer m_OutputTimeSelector;

    //##Description
    //## @brief Time when Header was last initialized
    itk::TimeStamp m_TimeOfHeaderInitialization;

    mitk::ScalarType m_OutsideValue;
    mitk::ScalarType m_MinValue;
    mitk::ScalarType m_MaxValue;
    bool m_OverrideOutsideValue;
  };

} // namespace mitk

#endif
