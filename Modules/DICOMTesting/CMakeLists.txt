if(BUILD_TESTING)

# clear variables from prior files.cmake
# Else CMake would use the content of these variables and would try to create tests (which are not present in DICOMTesting).
set(MODULE_TESTS)
set(MODULE_IMAGE_TESTS)
set(MODULE_SURFACE_TESTS)
set(MODULE_TESTIMAGE)
set(MODULE_TESTSURFACE)
set(MODULE_CUSTOM_TESTS)
set(H_FILES)
set(CPP_FILES)

# now create a new module only for testing purposes
MITK_CREATE_MODULE(
  DEPENDS MitkDICOM
  PACKAGE_DEPENDS
    PRIVATE DCMTK|config ITK|IOGDCM
)

mitk_check_module_dependencies(MODULES MitkDICOMTesting MISSING_DEPENDENCIES_VAR _missing_deps)

if(_missing_deps)
  message(STATUS "mitkDICOMTesting module helper applications won't be built. Missing: ${_missing_deps}")
else(_missing_deps)

  # dumps out image information
  add_executable(DumpDICOMMitkImage src/DumpDICOMMitkImage.cpp)
  mitk_use_modules(TARGET DumpDICOMMitkImage MODULES MitkDICOMTesting)

  # compares dumped out image information against reference dump
  add_executable(VerifyDICOMMitkImageDump src/VerifyDICOMMitkImageDump.cpp)
  mitk_use_modules(TARGET VerifyDICOMMitkImageDump MODULES MitkDICOMTesting)

  set_property(TARGET DumpDICOMMitkImage VerifyDICOMMitkImageDump PROPERTY FOLDER
    "${MITK_ROOT_FOLDER}/Modules/Tests")

  add_subdirectory(test)
endif()

endif()
