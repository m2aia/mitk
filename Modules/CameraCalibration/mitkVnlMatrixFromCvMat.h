/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef mitkVnlMatrixFromCvMat_h
#define mitkVnlMatrixFromCvMat_h

#include <vnl/vnl_matrix.h>
#include <opencv2/core.hpp>
#include <mitkObservable.h>
#include <mitkAlgorithm.h>
#include <mitkEndoMacros.h>

namespace mitk
{
  ///
  /// create a vnl_matrix from a cv mat
  ///
  template <class T>
  class VnlMatrixFromCvMat:
    virtual public Algorithm
  {
  public:
    ///
    /// init default values and save references
    ///
    VnlMatrixFromCvMat(
      const cv::Mat* _CvMat,
      vnl_matrix<T>* _VnlMatrix):
        m_CvMat(_CvMat),
        m_VnlMatrix(_VnlMatrix)
    {
    }


    ///
    /// cv mat to vnl matrix with known cv type
    ///
    template <typename TCvMat, typename TVnlMat>
    void ToVnlMatrix( vnl_matrix<TVnlMat>& vnlMat, const cv::Mat& mat )
    {
      vnlMat.set_size( mat.rows, mat.cols );
      for(int i=0; i<mat.rows; ++i)
        for(int j=0; j<mat.cols; ++j)
          vnlMat(i,j) = static_cast<TVnlMat>( mat.at<TCvMat>(i,j) );
    }

    ///
    /// executes the Algorithm
    ///
    void Update() override
    {
      endoAccessCvMat( ToVnlMatrix, T, (*m_VnlMatrix), (*m_CvMat) );
    }
  private:
    ///
    /// VnlMatrixFromCvMat input member variable
    ///
    const cv::Mat* m_CvMat;
    ///
    /// VnlMatrixFromCvMat output member variable
    ///
    vnl_matrix<T>* m_VnlMatrix;
  };
} // namespace mitk

#endif
