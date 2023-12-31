/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkGetClassHierarchy_h
#define mitkGetClassHierarchy_h

#include <string>
#include <vector>

namespace mitk
{
#ifndef DOXYGEN_SKIP

  template <typename T>
  class HasMethodGetStaticNameOfClass
  {
    typedef char Small;
    struct Big
    {
      char dummy[2];
    };

    template <const char *(*)()>
    struct SFINAE
    {
    };
    template <typename U>
    static Small Test(SFINAE<&U::GetStaticNameOfClass> *);
    template <typename U>
    static Big Test(...);

  public:
    enum
    {
      value = sizeof(Test<T>(nullptr)) == sizeof(Small)
    };
  };

  template <typename T, bool>
  struct StaticNameOfClass
  {
    static std::string value() { return typeid(T).name(); }
  };

  template <typename T>
  struct StaticNameOfClass<T, true>
  {
    static std::string value() { return T::GetStaticNameOfClass(); }
  };

  template <typename T>
  class HasTypeSuperclass
  {
    typedef char Small;
    struct Big
    {
      char dummy[2];
    };

    template <typename U>
    static Small Test(typename U::Superclass *);
    template <typename U>
    static Big Test(...);

  public:
    enum
    {
      value = sizeof(Test<T>(nullptr)) == sizeof(Small)
    };
  };

  template <typename T, bool>
  struct GetSuperclassType
  {
    typedef void value;
  };

  template <typename T>
  struct GetSuperclassType<T, true>
  {
    typedef typename T::Superclass value;
  };

  template <typename T>
  std::vector<std::string> GetClassHierarchy();

  template <>
  inline std::vector<std::string> GetClassHierarchy<void>()
  {
    return std::vector<std::string>();
  }

#endif

  /**
   * Get the class hierarchy for MITK classes as a list of class names.
   *
   * This function will return the name of classes and their direct
   * super-classes as specified in the mitkClassMacro. The order is from
   * most derived class to the last base class containing the mitkClassMacro.
   */
  template <typename T>
  std::vector<std::string> GetClassHierarchy()
  {
    std::vector<std::string> result;
    std::string name = StaticNameOfClass<T, HasMethodGetStaticNameOfClass<T>::value>::value();
    if (!name.empty())
      result.push_back(name);
    std::vector<std::string> superHierarchy =
      GetClassHierarchy<typename GetSuperclassType<T, HasTypeSuperclass<T>::value>::value>();
    result.insert(result.end(), superHierarchy.begin(), superHierarchy.end());
    return result;
  }
}

#endif
