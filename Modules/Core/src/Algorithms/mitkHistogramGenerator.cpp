/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#if (_MSC_VER == 1200)
#include <itkFixedCenterOfRotationAffineTransform.h>
#endif

#include "mitkHistogramGenerator.h"
#include "mitkImageAccessByItk.h"
#include "mitkImageTimeSelector.h"

//
// The new ITK Statistics framework has
// a class with the same functionality as
// MITKScalarImageToHistogramGenerator.h, but
// no longer has the classis the MITK class depends on.
#if !defined(ITK_USE_REVIEW_STATISTICS)
#include "itkMITKScalarImageToHistogramGenerator.h"
#else
#include "itkScalarImageToHistogramGenerator.h"
#endif

mitk::HistogramGenerator::HistogramGenerator() : m_Image(nullptr), m_Size(256), m_Histogram(nullptr)
{
}

mitk::HistogramGenerator::~HistogramGenerator()
{
}

template <typename TPixel, unsigned int VImageDimension>
void InternalCompute(itk::Image<TPixel, VImageDimension> *itkImage,
                     const mitk::HistogramGenerator *mitkHistoGenerator,
                     mitk::HistogramGenerator::HistogramType::ConstPointer &histogram)
{
#if !defined(ITK_USE_REVIEW_STATISTICS)
  typedef itk::Statistics::MITKScalarImageToHistogramGenerator<itk::Image<TPixel, VImageDimension>, double>
    HistogramGeneratorType;
#else
  typedef itk::Statistics::ScalarImageToHistogramGenerator<itk::Image<TPixel, VImageDimension>> HistogramGeneratorType;

#endif
  typename HistogramGeneratorType::Pointer histogramGenerator = HistogramGeneratorType::New();

  histogramGenerator->SetInput(itkImage);

  histogramGenerator->SetNumberOfBins(mitkHistoGenerator->GetSize());
  //  histogramGenerator->SetMarginalScale( 10.0 );
  histogramGenerator->Compute();

  histogram = histogramGenerator->GetOutput();
}

void mitk::HistogramGenerator::ComputeHistogram()
{
  if ((m_Histogram.IsNull()) || (m_Histogram->GetMTime() < m_Image->GetMTime()))
  {
    const_cast<mitk::Image *>(m_Image.GetPointer())->SetRequestedRegionToLargestPossibleRegion(); //@todo without this,
                                                                                                  // Image::GetScalarMin
    // does not work for
    // dim==3 (including
    // sliceselector!)
    const_cast<mitk::Image *>(m_Image.GetPointer())->Update();
    mitk::ImageTimeSelector::Pointer timeSelector = mitk::ImageTimeSelector::New();
    timeSelector->SetInput(m_Image);
    timeSelector->SetTimeNr(0);
    timeSelector->UpdateLargestPossibleRegion();
    AccessByItk_n(timeSelector->GetOutput(), InternalCompute, (this, m_Histogram));
  }

  // debug code
  /*
    MITK_INFO << "Histogram modified 1" << m_Histogram->GetMTime() << std::endl;
    m_Histogram->Modified();
    MITK_INFO << "Histogram modified 2" << m_Histogram->GetMTime() << std::endl;
    MITK_INFO << "Image modified" << m_Image->GetMTime() << std::endl;
  const unsigned int histogramSize = m_Histogram->Size();

  MITK_INFO << "Histogram size " << histogramSize << std::endl;

  HistogramType::ConstIterator itr = GetHistogram()->Begin();
  HistogramType::ConstIterator end = GetHistogram()->End();

  int bin = 0;
  while( itr != end )
    {
    MITK_INFO << "bin = " << GetHistogram()->GetBinMin(0,bin) << "--" << GetHistogram()->GetBinMax(0,bin) << " frequency
  = ";
    MITK_INFO << itr.GetFrequency() << std::endl;
    ++itr;
    ++bin;
    }
    */
}

float mitk::HistogramGenerator::GetMaximumFrequency() const
{
  return CalculateMaximumFrequency(this->m_Histogram);
}

float mitk::HistogramGenerator::CalculateMaximumFrequency(const HistogramType *histogram)
{
  HistogramType::ConstIterator itr = histogram->Begin();
  HistogramType::ConstIterator end = histogram->End();

  float maxFreq = 0;
  while (itr != end)
  {
    maxFreq = std::max(maxFreq, static_cast<float>(itr.GetFrequency()));
    ++itr;
  }
  return maxFreq;
}
