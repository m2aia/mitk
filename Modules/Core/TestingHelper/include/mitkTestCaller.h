/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkTestCaller_h
#define mitkTestCaller_h

#include "cppunit/TestCase.h"

#include <string>
#include <vector>

extern std::vector<std::string> globalCmdLineArgs;

namespace mitk
{
  /**
   * \brief A test caller for parameterized tests.
   *
   * This class is not meant to be used directly. Use the
   * mitk::TestFixture class and MITK_PARAMETERIZED_TEST
   * instead.
   */
  template <class ParameterizedFixture>
  class TestCaller : public CppUnit::TestCase
  {
    typedef void (ParameterizedFixture::*TestMethod)();

  public:
    /**
     * Constructor for TestCaller. This constructor builds a new ParameterizedFixture
     * instance owned by the TestCaller.
     * \param name name of this TestCaller
     * \param test the method this TestCaller calls in runTest()
     */
    TestCaller(const std::string &name, TestMethod test)
      : TestCase(name), m_OwnFixture(true), m_Fixture(new ParameterizedFixture()), m_Test(test)
    {
      m_Fixture->m_Parameter = globalCmdLineArgs;
    }

    /**
     * Constructor for TestCaller.
     * This constructor does not create a new ParameterizedFixture instance but accepts
     * an existing one as parameter. The TestCaller will not own the
     * ParameterizedFixture object.
     * \param name name of this TestCaller
     * \param test the method this TestCaller calls in runTest()
     * \param fixture the Fixture to invoke the test method on.
     */
    TestCaller(const std::string &name, TestMethod test, ParameterizedFixture &fixture)
      : TestCase(name), m_OwnFixture(false), m_Fixture(&fixture), m_Test(test)
    {
      m_Fixture->m_Parameter = globalCmdLineArgs;
    }

    /**
     * Constructor for TestCaller.
     * This constructor does not create a new ParameterizedFixture instance but accepts
     * an existing one as parameter. The TestCaller will own the
     * ParameterizedFixture object and delete it in its destructor.
     * \param name name of this TestCaller
     * \param test the method this TestCaller calls in runTest()
     * \param fixture the Fixture to invoke the test method on.
     */
    TestCaller(const std::string &name, TestMethod test, ParameterizedFixture *fixture)
      : TestCase(name), m_OwnFixture(true), m_Fixture(fixture), m_Test(test)
    {
      m_Fixture->m_Parameter = globalCmdLineArgs;
    }

    /**
     * Constructor for TestCaller.
     * This constructor does not create a new ParameterizedFixture instance but accepts
     * an existing one as parameter. The TestCaller will own the
     * ParameterizedFixture object and delete it in its destructor.
     * \param name name of this TestCaller
     * \param test the method this TestCaller calls in runTest()
     * \param fixture the Fixture to invoke the test method on.
     * \param param A list of string parameters for the fixture.
     */
    TestCaller(const std::string &name,
               TestMethod test,
               ParameterizedFixture *fixture,
               const std::vector<std::string> &param)
      : TestCase(name), m_OwnFixture(true), m_Fixture(fixture), m_Test(test)
    {
      m_Fixture->m_Parameter = param;
    }

    ~TestCaller() override
    {
      if (m_OwnFixture)
        delete m_Fixture;
    }

    void runTest() override { (m_Fixture->*m_Test)(); }
    void setUp() override { m_Fixture->setUp(); }
    void tearDown() override { m_Fixture->tearDown(); }
    std::string toString() const { return "TestCaller " + getName(); }
  private:
    TestCaller(const TestCaller &other);
    TestCaller &operator=(const TestCaller &other);

  private:
    bool m_OwnFixture;
    ParameterizedFixture *m_Fixture;
    TestMethod m_Test;
  };
}

#endif
