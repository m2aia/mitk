/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkImageSliceSelector_h
#define mitkImageSliceSelector_h

#include "mitkSubImageSelector.h"
#include <MitkCoreExports.h>

namespace mitk
{
  //##Documentation
  //## @brief Provides access to a slice of the input image
  //##
  //## If the input is generated by a ProcessObject, only the required data is
  //## requested.
  //## @ingroup Process
  class MITKCORE_EXPORT ImageSliceSelector : public SubImageSelector
  {
  public:
    mitkClassMacro(ImageSliceSelector, SubImageSelector);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

    itkGetConstMacro(SliceNr, int);
    itkSetMacro(SliceNr, int);

    itkGetConstMacro(TimeNr, int);
    itkSetMacro(TimeNr, int);

    itkGetConstMacro(ChannelNr, int);
    itkSetMacro(ChannelNr, int);

  protected:
    void GenerateOutputInformation() override;

    void GenerateInputRequestedRegion() override;

    void GenerateData() override;

    ImageSliceSelector();

    ~ImageSliceSelector() override;

    int m_SliceNr;

    int m_TimeNr;

    int m_ChannelNr;
  };

} // namespace mitk

#endif
