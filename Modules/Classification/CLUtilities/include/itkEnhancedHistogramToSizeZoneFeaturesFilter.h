/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         https://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#ifndef __itkEnhancedHistogramToSizeZoneFeaturesFilter_h
#define __itkEnhancedHistogramToSizeZoneFeaturesFilter_h

#include "itkHistogram.h"
#include "itkMacro.h"
#include "itkProcessObject.h"
#include "itkSimpleDataObjectDecorator.h"

namespace itk {
  namespace Statistics {
    /** \class EnhancedHistogramToSizeZoneFeaturesFilter
    *  \brief This class computes texture feature coefficients from a grey level
    * Zone-length matrix.
    *
    * By default, Zone length features are computed for each spatial
    * direction and then averaged afterward, so it is possible to access the
    * standard deviations of the texture features. These values give a clue as
    * to texture anisotropy. However, doing this is much more work, because it
    * involved computing one for each offset given. To compute a single matrix
    * using the first offset, call FastCalculationsOn(). If this is called,
    * then the texture standard deviations will not be computed (and will be set
    * to zero), but texture computation will be much faster.
    *
    * This class is templated over the input histogram type.
    *
    * Print references:
    * M. M. Galloway. Texture analysis using gray level Zone lengths. Computer
    * Graphics and Image Processing, 4:172-179, 1975.
    *
    * A. Chu, C. M. Sehgal, and J. F. Greenleaf. Use of gray value distribution of
    * Zone lengths for texture analysis.  Pattern Recognition Letters, 11:415-420,
    * 1990.
    *
    * B. R. Dasarathy and E. B. Holder. Image characterizations based on joint
    * gray-level Zone-length distributions. Pattern Recognition Letters, 12:490-502,
    * 1991.
    *
    * IJ article: https://hdl.handle.net/1926/1374
    *
    * \sa ScalarImageToSizeZoneFeaturesFilter
    * \sa ScalarImageToSizeZoneMatrixFilter
    * \sa EnhancedHistogramToSizeZoneFeaturesFilter
    *
    * \author: Nick Tustison
    * \ingroup ITKStatistics
    */

    template< typename THistogram >
    class EnhancedHistogramToSizeZoneFeaturesFilter : public ProcessObject
    {
    public:
      /** Standard typedefs */
      typedef EnhancedHistogramToSizeZoneFeaturesFilter     Self;
      typedef ProcessObject                          Superclass;
      typedef SmartPointer<Self>                     Pointer;
      typedef SmartPointer<const Self>               ConstPointer;

      /** Zone-time type information (and related methods). */
      itkTypeMacro( EnhancedHistogramToSizeZoneFeaturesFilter, ProcessObject );

      /** standard New() method support */
      itkNewMacro( Self );

      typedef THistogram                                      HistogramType;
      typedef typename HistogramType::Pointer                 HistogramPointer;
      typedef typename HistogramType::ConstPointer            HistogramConstPointer;
      typedef typename HistogramType::MeasurementType         MeasurementType;
      typedef typename HistogramType::MeasurementVectorType   MeasurementVectorType;
      typedef typename HistogramType::IndexType               IndexType;
      typedef typename HistogramType::
        TotalAbsoluteFrequencyType                            FrequencyType;

      /** Method to Set/Get the input Histogram */
      using Superclass::SetInput;
      void SetInput ( const HistogramType * histogram );
      const HistogramType * GetInput() const;

      /** Smart Pointer type to a DataObject. */
      typedef DataObject::Pointer                   DataObjectPointer;

      /** Type of DataObjects used for scalar outputs */
      typedef SimpleDataObjectDecorator<MeasurementType>     MeasurementObjectType;

      /** Methods to return the short Zone emphasis. */
      MeasurementType GetSmallZoneEmphasis() const;
      const MeasurementObjectType* GetSmallZoneEmphasisOutput() const;

      /** Methods to return the long Zone emphasis. */
      MeasurementType GetLargeZoneEmphasis() const;
      const MeasurementObjectType* GetLargeZoneEmphasisOutput() const;

      /** Methods to return the grey level nonuniformity. */
      MeasurementType GetGreyLevelNonuniformity() const;
      const MeasurementObjectType* GetGreyLevelNonuniformityOutput() const;

      /** Methods to return the grey level nonuniformity normalized. */
      MeasurementType GetGreyLevelNonuniformityNormalized() const;
      const MeasurementObjectType* GetGreyLevelNonuniformityNormalizedOutput() const;

      /** Methods to return the Zone length nonuniformity. */
      MeasurementType GetSizeZoneNonuniformity() const;
      const MeasurementObjectType* GetSizeZoneNonuniformityOutput() const;

      /** Methods to return the Zone length nonuniformity normalized. */
      MeasurementType GetSizeZoneNonuniformityNormalized() const;
      const MeasurementObjectType* GetSizeZoneNonuniformityNormalizedOutput() const;

      /** Methods to return the low grey level Zone emphasis. */
      MeasurementType GetLowGreyLevelZoneEmphasis() const;
      const MeasurementObjectType* GetLowGreyLevelZoneEmphasisOutput() const;

      /** Methods to return the high grey level Zone emphasis. */
      MeasurementType GetHighGreyLevelZoneEmphasis() const;
      const MeasurementObjectType* GetHighGreyLevelZoneEmphasisOutput() const;

      /** Methods to return the short Zone low grey level Zone emphasis. */
      MeasurementType GetSmallZoneLowGreyLevelEmphasis() const;
      const MeasurementObjectType* GetSmallZoneLowGreyLevelEmphasisOutput() const;

      /** Methods to return the short Zone high grey level Zone emphasis. */
      MeasurementType GetSmallZoneHighGreyLevelEmphasis() const;
      const MeasurementObjectType* GetSmallZoneHighGreyLevelEmphasisOutput() const;

      /** Methods to return the long Zone low grey level Zone emphasis. */
      MeasurementType GetLargeZoneLowGreyLevelEmphasis() const;
      const MeasurementObjectType* GetLargeZoneLowGreyLevelEmphasisOutput() const;

      /** Methods to return the long Zone high grey level Zone emphasis. */
      MeasurementType GetLargeZoneHighGreyLevelEmphasis() const;
      const MeasurementObjectType* GetLargeZoneHighGreyLevelEmphasisOutput() const;

      /** Methods to return the long Zone high grey level Zone emphasis. */
      MeasurementType GetZonePercentage() const;
      const MeasurementObjectType* GetZonePercentageOutput() const;

      /** Methods to return the long Zone high grey level Zone emphasis. */
      MeasurementType GetNumberOfZones() const;
      const MeasurementObjectType* GetNumberOfZonesOutput() const;

      /** Methods to return the grey level variance. */
      MeasurementType GetGreyLevelVariance() const;
      const MeasurementObjectType* GetGreyLevelVarianceOutput() const;

      /** Methods to return the Zone length variance. */
      MeasurementType GetSizeZoneVariance() const;
      const MeasurementObjectType* GetSizeZoneVarianceOutput() const;

      /** Methods to return the Zone entropy. */
      MeasurementType GetZoneEntropy() const;
      const MeasurementObjectType* GetZoneEntropyOutput() const;

      itkGetMacro( TotalNumberOfZones, unsigned long );

      itkGetConstMacro(NumberOfVoxels, unsigned long);
      itkSetMacro(NumberOfVoxels, unsigned long);

      /** Zone-length feature types */
      typedef enum
      {
        SmallZoneEmphasis,
        LargeZoneEmphasis,
        GreyLevelNonuniformity,
        GreyLevelNonuniformityNormalized,
        SizeZoneNonuniformity,
        SizeZoneNonuniformityNormalized,
        LowGreyLevelZoneEmphasis,
        HighGreyLevelZoneEmphasis,
        SmallZoneLowGreyLevelEmphasis,
        SmallZoneHighGreyLevelEmphasis,
        LargeZoneLowGreyLevelEmphasis,
        LargeZoneHighGreyLevelEmphasis,
        ZonePercentage,
        GreyLevelVariance,
        SizeZoneVariance,
        ZoneEntropy
      }  SizeZoneFeatureName;

      /** convenience method to access the Zone length values */
      MeasurementType GetFeature( SizeZoneFeatureName name );

    protected:
      EnhancedHistogramToSizeZoneFeaturesFilter();
      ~EnhancedHistogramToSizeZoneFeaturesFilter() {};
      virtual void PrintSelf(std::ostream& os, Indent indent) const ITK_OVERRIDE;

      /** Make a DataObject to be used for output output. */
      typedef ProcessObject::DataObjectPointerArraySizeType DataObjectPointerArraySizeType;
      using Superclass::MakeOutput;
      virtual DataObjectPointer MakeOutput( DataObjectPointerArraySizeType ) ITK_OVERRIDE;

      virtual void GenerateData() ITK_OVERRIDE;

    private:
      EnhancedHistogramToSizeZoneFeaturesFilter(const Self&); //purposely not implemented
      void operator=(const Self&); //purposely not implemented

      unsigned long                           m_TotalNumberOfZones;
      unsigned long                           m_NumberOfVoxels;
    };
  } // end of namespace Statistics
} // end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkEnhancedHistogramToSizeZoneFeaturesFilter.hxx"
#endif

#endif
