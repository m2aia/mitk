/*============================================================================

  Library: CppMicroServices

  Copyright (c) German Cancer Research Center (DKFZ)
  All rights reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

============================================================================*/

#include <usLDAPFilter.h>
#include <usLDAPProp.h>

#include "usTestingMacros.h"

#include <stdexcept>

US_USE_NAMESPACE

int TestParsing()
{
  // WELL FORMED Expr
  try
  {
    US_TEST_OUTPUT(<< "Parsing (cn=Babs Jensen)")
    LDAPFilter ldap( "(cn=Babs Jensen)" );
    US_TEST_OUTPUT(<< "Parsing (!(cn=Tim Howes))")
    ldap = LDAPFilter( "(!(cn=Tim Howes))" );
    US_TEST_OUTPUT(<< "Parsing " << std::string("(&(") + ServiceConstants::OBJECTCLASS() + "=Person)(|(sn=Jensen)(cn=Babs J*)))")
    ldap = LDAPFilter( std::string("(&(") + ServiceConstants::OBJECTCLASS() + "=Person)(|(sn=Jensen)(cn=Babs J*)))" );
    US_TEST_OUTPUT(<< "Parsing (o=univ*of*mich*)")
    ldap = LDAPFilter( "(o=univ*of*mich*)" );
  }
  catch (const std::invalid_argument& e)
  {
    US_TEST_OUTPUT(<< e.what());
    return EXIT_FAILURE;
  }


  // MALFORMED Expr
  try
  {
    US_TEST_OUTPUT( << "Parsing malformed expr: cn=Babs Jensen)")
    LDAPFilter ldap( "cn=Babs Jensen)" );
    return EXIT_FAILURE;
  }
  catch (const std::invalid_argument&)
  {
  }

  return EXIT_SUCCESS;
}


int TestEvaluate()
{
  // EVALUATE
  try
  {
    LDAPFilter ldap( "(Cn=Babs Jensen)" );
    ServiceProperties props;
    bool eval = false;

    // Several values
    props["cn"] = std::string("Babs Jensen");
    props["unused"] = std::string("Jansen");
    US_TEST_OUTPUT(<< "Evaluating expr: " << ldap.ToString())
    eval = ldap.Match(props);
    if (!eval)
    {
      return EXIT_FAILURE;
    }

    // WILDCARD
    ldap = LDAPFilter( "(cn=Babs *)" );
    props.clear();
    props["cn"] = std::string("Babs Jensen");
    US_TEST_OUTPUT(<< "Evaluating wildcard expr: " << ldap.ToString())
    eval = ldap.Match(props);
    if ( !eval )
    {
      return EXIT_FAILURE;
    }

    // NOT FOUND
    ldap = LDAPFilter( "(cn=Babs *)" );
    props.clear();
    props["unused"] = std::string("New");
    US_TEST_OUTPUT(<< "Expr not found test: " << ldap.ToString())
    eval = ldap.Match(props);
    if ( eval )
    {
      return EXIT_FAILURE;
    }

    // std::vector with integer values
    ldap = LDAPFilter( "  ( |(cn=Babs *)(sn=1) )" );
    props.clear();
    std::vector<Any> list;
    list.push_back(std::string("Babs Jensen"));
    list.push_back(std::string("1"));
    props["sn"] = list;
    US_TEST_OUTPUT(<< "Evaluating vector expr: " << ldap.ToString())
    eval = ldap.Match(props);
    if (!eval)
    {
      return EXIT_FAILURE;
    }

    // wrong case
    ldap = LDAPFilter( "(cN=Babs *)" );
    props.clear();
    props["cn"] = std::string("Babs Jensen");
    US_TEST_OUTPUT(<< "Evaluating case sensitive expr: " << ldap.ToString())
    eval = ldap.MatchCase(props);
    if (eval)
    {
      return EXIT_FAILURE;
    }
  }
  catch (const std::invalid_argument& e)
  {
    US_TEST_OUTPUT( << e.what() )
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

void TestLDAPExpressions()
{
  LDAPFilter filter(
        LDAPProp("bla") == "jo" && !(LDAPProp("ha") == 1) &&
        (LDAPProp("presence") || !LDAPProp("absence")) &&
        LDAPProp("le") <= 4.1 && LDAPProp("ge") >= -3 &&
        LDAPProp("approx").Approx("Approx")
        );
  const std::string filterStr = "(&(&(&(&(&(bla=jo)(!(ha=1)))(|(presence=*)(!(absence=*))))(le<=4.1))(ge>=-3))(approx~=Approx))";
  US_TEST_CONDITION(filter.ToString() == filterStr, "test generated filter string")

  std::string emptyValue;
  std::string someValue = "some";
  std::string filter1 = LDAPProp("key2") == someValue && LDAPProp("key3");
  std::string filter2 = LDAPProp("key2") == someValue && (LDAPProp("key1") == emptyValue || LDAPProp("key3"));
  US_TEST_CONDITION(filter1 == filter2, "test null expressions")
}

int usLDAPFilterTest(int /*argc*/, char* /*argv*/[])
{
  US_TEST_BEGIN("LDAPFilterTest");

  TestLDAPExpressions();
  US_TEST_CONDITION(TestParsing() == EXIT_SUCCESS, "Parsing LDAP expressions: ")
  US_TEST_CONDITION(TestEvaluate() == EXIT_SUCCESS, "Evaluating LDAP expressions: ")

  US_TEST_END()
}
