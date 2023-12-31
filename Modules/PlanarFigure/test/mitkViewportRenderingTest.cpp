/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

// MITK
#include "mitkColorProperty.h"
#include "mitkImage.h"
#include "mitkLevelWindowProperty.h"
#include "mitkLevelWindowProperty.h"
#include "mitkNodePredicateDataType.h"
#include "mitkPlanarFigure.h"
#include "mitkRenderingTestHelper.h"
#include "mitkSurface.h"
#include "mitkTestingMacros.h"
#include <mitkTestNotRunException.h>

// ITK
#include <itkVectorContainer.h>

// VTK
#include <vtkDebugLeaks.h>
#include <vtkRegressionTestImage.h>

// stdlib
#include <cstdlib>

int mitkViewportRenderingTest(int argc, char *argv[])
{
  try
  {
    mitk::RenderingTestHelper openGlTest(640, 480);
  }
  catch (const mitk::TestNotRunException &e)
  {
    MITK_WARN << "Test not run: " << e.GetDescription();
    return 77;
  }

  // load all arguments into a datastorage, take last argument as reference rendering
  // setup a renderwindow of fixed size X*Y
  // render the datastorage
  // compare rendering to reference image
  MITK_TEST_BEGIN("mitkViewportRenderingTest")

  /// \todo Fix leaks of vtkObjects. Bug 18095.
  vtkDebugLeaks::SetExitError(0);

  // enough parameters?
  if (argc < 2)
  {
    MITK_TEST_OUTPUT(<< "Usage: " << std::string(*argv) << " [file1 file2 ...] outputfile")
    MITK_TEST_OUTPUT(<< "Will render a central axial slice of all given files into outputfile")
    exit(EXIT_FAILURE);
  }

  double renderWindowWidth = atof(argv[1]);
  double renderWindowHeight = atof(argv[2]);
  double left = atof(argv[3]);
  double bottom = atof(argv[4]);
  double right = atof(argv[5]);
  double top = atof(argv[6]);
  std::string referenceFilename = argv[8 + 6];
  argv += 6; // DO NOT attempt to read these as files, this just makes no sense
  argc -= 6; // DO NOT attempt to read these as files, this just makes no sense

  MITK_INFO << "Testing viewport " << right - left << "x" << top - bottom << " (" << left << ", " << bottom << ") to ("
            << right << ", " << top << ") "
            << "in render window of size " << renderWindowWidth << "x" << renderWindowHeight << "px";

  mitk::RenderingTestHelper renderingHelper(renderWindowWidth, renderWindowHeight, argc, argv); // non-power-of-2
  renderingHelper.SetViewDirection(mitk::AnatomicalPlane::Axial);

  typedef mitk::DataStorage::SetOfObjects ObjectsSet;

  ObjectsSet::ConstPointer figures =
    renderingHelper.GetDataStorage()->GetSubset(mitk::TNodePredicateDataType<mitk::PlanarFigure>::New());
  for (auto iterFigures = figures->begin(); iterFigures != figures->end(); ++iterFigures)
  {
    (*iterFigures)->SetProperty("planarfigure.default.line.color", mitk::ColorProperty::New(1.0, 0.0, 0.0)); // red
    (*iterFigures)->SetProperty("planarfigure.drawcontrolpoints", mitk::BoolProperty::New(false));
    (*iterFigures)->SetProperty("planarfigure.drawname", mitk::BoolProperty::New(false));
    (*iterFigures)->SetProperty("planarfigure.drawquantities", mitk::BoolProperty::New(true));
  }

  ObjectsSet::ConstPointer surfaces =
    renderingHelper.GetDataStorage()->GetSubset(mitk::TNodePredicateDataType<mitk::Surface>::New());
  for (auto iterSurfaces = surfaces->begin(); iterSurfaces != surfaces->end(); ++iterSurfaces)
  {
    (*iterSurfaces)->SetProperty("color", mitk::ColorProperty::New(0.0, 1.0, 0.0)); // green
  }

  ObjectsSet::ConstPointer images =
    renderingHelper.GetDataStorage()->GetSubset(mitk::TNodePredicateDataType<mitk::Image>::New());
  for (auto iterImages = images->begin(); iterImages != images->end(); ++iterImages)
  {
    (*iterImages)->SetProperty("levelwindow", mitk::LevelWindowProperty::New(mitk::LevelWindow(128.0, 256.0))); // green
    int imageWidth = dynamic_cast<mitk::Image *>((*iterImages)->GetData())->GetDimension(0);
    int imageHeight = dynamic_cast<mitk::Image *>((*iterImages)->GetData())->GetDimension(1);
    MITK_INFO << "Image dimension " << imageWidth << "x" << imageHeight;
  }

  double vLeft = left / renderWindowWidth;
  double vBottom = bottom / renderWindowHeight;
  double vRight = right / renderWindowWidth;
  double vTop = top / renderWindowHeight;
  // THIS HERE IS THE ACTUAL TEST PART, all the rest is setup and decoration

  mitk::RenderingManager::GetInstance()->InitializeViews(
    renderingHelper.GetDataStorage()->ComputeBoundingGeometry3D(images));

  renderingHelper.GetVtkRenderer()->SetViewport(vLeft, vBottom, vRight, vTop);
  renderingHelper.SetAutomaticallyCloseRenderWindow(true); // set to false for testing the test itself
  renderingHelper.Render();
  // use this to generate a reference screenshot or save the file:
  bool generateReferenceScreenshot = false;
  if (generateReferenceScreenshot)
  {
    std::string tmpFilename = referenceFilename;
    std::string::size_type slashpos = referenceFilename.find_last_of('/');
    tmpFilename = referenceFilename.substr(slashpos + 1);
    tmpFilename = std::string("/tmp/") + tmpFilename;
    renderingHelper.SaveAsPNG(tmpFilename);
    MITK_INFO << "*********************************";
    MITK_INFO << "SAVE TO " << tmpFilename;
    MITK_INFO << "*********************************";
  }

  //### Usage of vtkRegressionTestImage:
  // vtkRegressionTestImage( vtkRenderWindow )
  // Set a vtkRenderWindow containing the desired scene.
  // vtkRegressionTestImage automatically searches in argc and argv[]
  // for a path a valid image with -V. If the test failed with the
  // first image (foo.png) check if there are images of the form
  // foo_N.png (where N=1,2,3...) and compare against them.
  int retVal = vtkRegressionTestImageThreshold(renderingHelper.GetVtkRenderWindow(), 20.0);

  // retVal meanings: (see VTK/Rendering/vtkTesting.h)
  // 0 = test failed
  // 1 = test passed
  // 2 = test not run
  // 3 = something with vtkInteraction
  MITK_TEST_CONDITION(retVal == 1, "VTK rendering result matches expectation");

  MITK_TEST_END();
}
