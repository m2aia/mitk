/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

// MITK HEADER
#include "mitkUSImageVideoSource.h"
#include "mitkImage.h"

//Other
#include <cstdio>
#include <opencv2/videoio/legacy/constants_c.h>

mitk::USImageVideoSource::USImageVideoSource()
  : m_VideoCapture(new cv::VideoCapture()),
  m_IsVideoReady(false),
  m_IsGreyscale(false),
  m_IsCropped(false),
  m_ResolutionOverrideWidth(0),
  m_ResolutionOverrideHeight(0),
  m_ResolutionOverride(false),
  m_GrayscaleFilter(mitk::ConvertGrayscaleOpenCVImageFilter::New()),
  m_CropFilter(mitk::CropOpenCVImageFilter::New())
{
}

mitk::USImageVideoSource::~USImageVideoSource()
{
  m_VideoCapture->release();
  delete m_VideoCapture;
}

void mitk::USImageVideoSource::SetVideoFileInput(std::string path)
{
  m_VideoCapture->open(path.c_str());

  // check if we succeeded
  if(!m_VideoCapture->isOpened()) { m_IsVideoReady = false; }
  else { m_IsVideoReady = true; }

  // if Override is enabled, use it
  if (m_ResolutionOverride)
  {
    m_VideoCapture->set(CV_CAP_PROP_FRAME_WIDTH, this->m_ResolutionOverrideWidth);
    m_VideoCapture->set(CV_CAP_PROP_FRAME_HEIGHT, this->m_ResolutionOverrideHeight);
  }
}

void mitk::USImageVideoSource::SetCameraInput(int deviceID)
{
  m_VideoCapture->open(deviceID);
  if(!m_VideoCapture->isOpened())  // check if we succeeded
    m_IsVideoReady = false;
  else
    m_IsVideoReady = true;

  // if Override is enabled, use it
  if (m_ResolutionOverride)
  {
    m_VideoCapture->set(CV_CAP_PROP_FRAME_WIDTH, this->m_ResolutionOverrideWidth);
    m_VideoCapture->set(CV_CAP_PROP_FRAME_HEIGHT, this->m_ResolutionOverrideHeight);
  }
}

void mitk::USImageVideoSource::ReleaseInput()
{
  m_VideoCapture->release();
  delete m_VideoCapture;
  m_VideoCapture = new cv::VideoCapture();
}

void mitk::USImageVideoSource::SetColorOutput(bool isColor){
  if ( ! isColor && ! m_IsGreyscale )
  {
    this->PushFilter(m_GrayscaleFilter.GetPointer());
  }
  else if ( isColor && m_IsGreyscale )
  {
    this->RemoveFilter(m_GrayscaleFilter.GetPointer());
  }

  m_IsGreyscale = !isColor;
}

int mitk::USImageVideoSource::GetImageHeight()
{
  if (m_VideoCapture) { return m_VideoCapture->get(CV_CAP_PROP_FRAME_HEIGHT); }
  else { return 0; }
}

int mitk::USImageVideoSource::GetImageWidth()
{
  if (m_VideoCapture) { return m_VideoCapture->get(CV_CAP_PROP_FRAME_WIDTH); }
  else { return 0; }
}

bool mitk::USImageVideoSource::GetIsReady()
{
  if (!m_VideoCapture) { return false; }

  return m_VideoCapture->isOpened();
}

void mitk::USImageVideoSource::SetRegionOfInterest(int topLeftX, int topLeftY, int bottomRightX, int bottomRightY)
{
  m_CropFilter->SetCropRegion(topLeftX, topLeftY, bottomRightX, bottomRightY);

  if (! m_IsCropped && ! m_CropFilter->GetIsCropRegionEmpty())
  {
    this->PushFilter(m_CropFilter.GetPointer());
    m_IsCropped = true;
  }
}

void mitk::USImageVideoSource::SetRegionOfInterest(USImageRoi roi)
{
  this->SetRegionOfInterest(roi.topLeftX, roi.topLeftY, roi.bottomRightX, roi.bottomRightY);
}

void mitk::USImageVideoSource::SetCropping(USImageCropping cropping)
{
  int width = this->GetImageWidth();
  int height = this->GetImageHeight();

  this->SetRegionOfInterest(cropping.left, cropping.top, width - cropping.right, height - cropping.bottom);
}

mitk::USImageVideoSource::USImageCropping mitk::USImageVideoSource::GetCropping()
{
  cv::Rect cropRect = m_CropFilter->GetCropRegion();

  USImageCropping cropping;
  cropping.left = cropRect.x;
  cropping.top = cropRect.y;

  if ( cropRect.height == 0 )
  {
    cropping.bottom = 0;
  }
  else
  {
    cropping.bottom = this->GetImageHeight() - (cropRect.y + cropRect.height);
  }

  if ( cropRect.width == 0 )
  {
    cropping.right = 0;
  }
  else
  {
    cropping.right = this->GetImageWidth() - (cropRect.x + cropRect.width);
  }

  return cropping;
}

mitk::USImageVideoSource::USImageRoi  mitk::USImageVideoSource::GetRegionOfInterest()
{
  cv::Rect cropRect = m_CropFilter->GetCropRegion();

  return USImageRoi(cropRect.x, cropRect.y, cropRect.x + cropRect.width, cropRect.y + cropRect.height);
}

void mitk::USImageVideoSource::RemoveRegionOfInterest()
{
  this->RemoveFilter(m_CropFilter.GetPointer());
  m_IsCropped = false;
}

void mitk::USImageVideoSource::GetNextRawImage(std::vector<cv::Mat>& image )
{
  // loop video if necessary
  //Commented out because setting and getting of these properties is not supported. Therefore on Linux
    //you'll always get some Highgui errors from OpenCV
  /*if (m_VideoCapture->get(CV_CAP_PROP_POS_FRAMES) == m_VideoCapture->get(CV_CAP_PROP_FRAME_COUNT))
  {
    m_VideoCapture->set(CV_CAP_PROP_POS_FRAMES, 0);
  }*/

  if (image.size() != 1)
    image.resize(1);

  // retrieve image
  *m_VideoCapture >> image[0]; // get a new frame from camera
}

void mitk::USImageVideoSource::GetNextRawImage(std::vector<mitk::Image::Pointer>& image )
{
  if (image.size() != 1)
    image.resize(1);

  std::vector<cv::Mat> cv_img;

  this->GetNextRawImage(cv_img);

  // convert to MITK-Image
  IplImage ipl_img = cvIplImage(cv_img[0]);

  this->m_OpenCVToMitkFilter->SetOpenCVImage(&ipl_img);
  this->m_OpenCVToMitkFilter->Update();

  // OpenCVToMitkImageFilter returns a standard mitk::image. We then transform it into an USImage
  image[0] = this->m_OpenCVToMitkFilter->GetOutput();

  // clean up
  cv_img[0].release();
}

void mitk::USImageVideoSource::OverrideResolution(int width, int height)
{
  this->m_ResolutionOverrideHeight = height;
  this->m_ResolutionOverrideWidth = width;

  if (m_VideoCapture != nullptr)
  {
    m_VideoCapture->set(CV_CAP_PROP_FRAME_WIDTH, width);
    m_VideoCapture->set(CV_CAP_PROP_FRAME_HEIGHT, height);
  }
}
