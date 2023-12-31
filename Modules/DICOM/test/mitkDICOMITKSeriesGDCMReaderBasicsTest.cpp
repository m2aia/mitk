/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkDICOMITKSeriesGDCMReader.h"
#include "mitkDICOMFileReaderTestHelper.h"
#include "mitkDICOMFilenameSorter.h"
#include "mitkDICOMTagBasedSorter.h"
#include "mitkDICOMSortByTag.h"

#include "mitkTestingMacros.h"

#include <unordered_map>
#include "mitkStringProperty.h"

using mitk::DICOMTag;

int mitkDICOMITKSeriesGDCMReaderBasicsTest(int argc, char* argv[])
{
  MITK_TEST_BEGIN("mitkDICOMITKSeriesGDCMReaderBasicsTest");

  mitk::DICOMITKSeriesGDCMReader::Pointer gdcmReader = mitk::DICOMITKSeriesGDCMReader::New();
  MITK_TEST_CONDITION_REQUIRED(gdcmReader.IsNotNull(), "DICOMITKSeriesGDCMReader can be instantiated.");

  mitk::DICOMFileReaderTestHelper::SetTestInputFilenames( argc,argv );

  std::unordered_map<std::string, std::string> expectedPropertyTypes;
  mitk::DICOMITKSeriesGDCMReader::AdditionalTagsMapType additionalTags;
  additionalTags.insert( std::make_pair(DICOMTag( 0x0008, 0x0005 ), "Test1") );
  additionalTags.insert( std::make_pair(DICOMTag( 0x0008, 0x0060 ), "Test2") );
  additionalTags.insert( std::make_pair(DICOMTag( 0x0020, 0x1041 ), "Test3") );

  gdcmReader->SetAdditionalTagsOfInterest( additionalTags );

  // check the Set/GetInput function
  mitk::DICOMFileReaderTestHelper::TestInputFilenames( gdcmReader );

  // check that output is a good reproduction of input (no duplicates, no new elements)
  mitk::DICOMFileReaderTestHelper::TestOutputsContainInputs( gdcmReader );


  // repeat test with filename based sorter in-between
  mitk::DICOMFilenameSorter::Pointer filenameSorter = mitk::DICOMFilenameSorter::New();
  gdcmReader->AddSortingElement( filenameSorter );
  mitk::DICOMFileReaderTestHelper::TestOutputsContainInputs( gdcmReader );

  // repeat test with some more realistic sorting
  gdcmReader = mitk::DICOMITKSeriesGDCMReader::New(); // this also tests destruction
  mitk::DICOMTagBasedSorter::Pointer tagSorter = mitk::DICOMTagBasedSorter::New();

  // all the things that split by tag in DicomSeriesReader
  tagSorter->AddDistinguishingTag( DICOMTag(0x0028, 0x0010) ); // Number of Rows
  tagSorter->AddDistinguishingTag( DICOMTag(0x0028, 0x0011) ); // Number of Columns
  tagSorter->AddDistinguishingTag( DICOMTag(0x0028, 0x0030) ); // Pixel Spacing
  tagSorter->AddDistinguishingTag( DICOMTag(0x0018, 0x1164) ); // Imager Pixel Spacing
  tagSorter->AddDistinguishingTag( DICOMTag(0x0020, 0x0037), new mitk::DICOMTagBasedSorter::CutDecimalPlaces(5) ); // Image Orientation (Patient)
  tagSorter->AddDistinguishingTag( DICOMTag(0x0020, 0x000e) ); // Series Instance UID
  tagSorter->AddDistinguishingTag( DICOMTag(0x0018, 0x0050) ); // Slice Thickness
  tagSorter->AddDistinguishingTag( DICOMTag(0x0028, 0x0008) ); // Number of Frames
  tagSorter->AddDistinguishingTag( DICOMTag(0x0020, 0x0052) ); // Frame of Reference UID

  // a sorter...
  mitk::DICOMSortCriterion::ConstPointer sorting =
    mitk::DICOMSortByTag::New( DICOMTag(0x0020, 0x0013), // instance number
      mitk::DICOMSortByTag::New( DICOMTag(0x0020, 0x0012), // acquisition number
        mitk::DICOMSortByTag::New( DICOMTag(0x0008, 0x0032), // acquisition time
          mitk::DICOMSortByTag::New( DICOMTag(0x0018, 0x1060), // trigger time
            mitk::DICOMSortByTag::New( DICOMTag(0x0008, 0x0018) // SOP instance UID (last resort, not really meaningful but decides clearly)
            ).GetPointer()
          ).GetPointer()
        ).GetPointer()
      ).GetPointer()
    ).GetPointer();
  tagSorter->SetSortCriterion( sorting );

  gdcmReader->AddSortingElement( tagSorter );

  gdcmReader->SetAdditionalTagsOfInterest( additionalTags );


  mitk::DICOMFileReaderTestHelper::TestOutputsContainInputs( gdcmReader );

  gdcmReader->PrintOutputs(std::cout, true);

  // really load images
  mitk::DICOMFileReaderTestHelper::TestMitkImagesAreLoaded( gdcmReader, additionalTags, expectedPropertyTypes );


  //////////////////////////////////////////////////////////////////////////
  //
  // Load the images again with another TagLookupTableToPropertyFunctor
  //
  //////////////////////////////////////////////////////////////////////////

  gdcmReader->SetTagLookupTableToPropertyFunctor( []( const mitk::DICOMCachedValueLookupTable& table )
  {
    return static_cast<mitk::BaseProperty::Pointer>( mitk::StringProperty::New( table.GetTableValue(0).Value ) );
  } );

  expectedPropertyTypes.insert(std::make_pair("Test1", "StringProperty"));
  expectedPropertyTypes.insert(std::make_pair("Test2", "StringProperty"));
  expectedPropertyTypes.insert(std::make_pair("Test3", "StringProperty"));

  mitk::DICOMFileReaderTestHelper::TestOutputsContainInputs( gdcmReader );

  gdcmReader->PrintOutputs(std::cout, true);

  // really load images
  mitk::DICOMFileReaderTestHelper::TestMitkImagesAreLoaded( gdcmReader, additionalTags, expectedPropertyTypes );


  MITK_TEST_END();
}
