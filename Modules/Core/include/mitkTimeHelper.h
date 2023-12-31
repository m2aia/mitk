/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkTimeHelper_h
#define mitkTimeHelper_h

#include <mitkTimeGeometry.h>

namespace mitk
{
  //## @brief convert the start- and end-index-time of output-region in
  //## start- and end-index-time of input-region via millisecond-time
  template <class TOutputRegion, class TInputRegion>
  void GenerateTimeInInputRegion(const mitk::TimeGeometry *outputTimeGeometry,
                                 const TOutputRegion &outputRegion,
                                 const mitk::TimeGeometry *inputTimeGeometry,
                                 TInputRegion &inputRegion)
  {
    assert(outputTimeGeometry != nullptr);
    assert(inputTimeGeometry != nullptr);

    // convert the start-index-time of output in start-index-time of input via millisecond-time
    mitk::TimePointType timeInMS = outputTimeGeometry->TimeStepToTimePoint(outputRegion.GetIndex(3));
    mitk::TimeStepType timestep = inputTimeGeometry->TimePointToTimeStep(timeInMS);
    if ((timeInMS > itk::NumericTraits<mitk::ScalarType>::NonpositiveMin()) &&
        (inputTimeGeometry->IsValidTimeStep(timestep)))
      inputRegion.SetIndex(3, timestep);
    else
      inputRegion.SetIndex(3, 0);
    // convert the end-index-time of output in end-index-time of input via millisecond-time
    timeInMS = outputTimeGeometry->TimeStepToTimePoint(outputRegion.GetIndex(3) + outputRegion.GetSize(3) - 1);
    timestep = inputTimeGeometry->TimePointToTimeStep(timeInMS);
    if ((timeInMS > itk::NumericTraits<mitk::ScalarType>::NonpositiveMin()) &&
        (outputTimeGeometry->IsValidTimeStep(timestep)))
      inputRegion.SetSize(3, timestep - inputRegion.GetIndex(3) + 1);
    else
      inputRegion.SetSize(3, 1);
  }

  //##Documentation
  //## @brief convert the start- and end-index-time of output in
  //## start- and end-index-time of input1 and input2 via millisecond-time
  template <class TOutputData, class TInputData>
  void GenerateTimeInInputRegion(const TOutputData *output, TInputData *input)
  {
    assert(output != nullptr);
    assert(input != nullptr);

    const typename TOutputData::RegionType &outputRegion = output->GetRequestedRegion();
    typename TInputData::RegionType inputRegion;

    if (outputRegion.GetSize(3) < 1)
    {
      typename TInputData::RegionType::SizeType inputsize;
      inputsize.Fill(0);
      inputRegion.SetSize(inputsize);
      input->SetRequestedRegion(&inputRegion);
    }

    // convert the start-index-time of output in start-index-time of input via millisecond-time
    inputRegion = input->GetRequestedRegion();
    GenerateTimeInInputRegion(output->GetTimeGeometry(), outputRegion, input->GetTimeGeometry(), inputRegion);
    input->SetRequestedRegion(&inputRegion);
  }

} // end namespace mitk

#endif
