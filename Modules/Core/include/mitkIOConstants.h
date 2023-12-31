/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkIOConstants_h
#define mitkIOConstants_h

#include <MitkCoreExports.h>

#include <string>

namespace mitk
{
  /**
   * @ingroup IO
   * @brief The IOConstants struct
   */
  struct MITKCORE_EXPORT IOConstants
  {
    static std::string PIXEL_TYPE();
    static std::string PIXEL_TYPE_CHAR();
    static std::string PIXEL_TYPE_UCHAR();
    static std::string PIXEL_TYPE_SHORT();
    static std::string PIXEL_TYPE_USHORT();
    static std::string PIXEL_TYPE_INT();
    static std::string PIXEL_TYPE_UINT();
    static std::string PIXEL_TYPE_FLOAT();
    static std::string PIXEL_TYPE_DOUBLE();
    static std::string PIXEL_TYPE_ENUM();

    static std::string DIMENSION();
    static std::string DIMENSION_ENUM();

    static std::string ENDIANNESS();
    static std::string ENDIANNESS_LITTLE();
    static std::string ENDIANNESS_BIG();
    static std::string ENDIANNESS_ENUM();

    static std::string SIZE_X();
    static std::string SIZE_Y();
    static std::string SIZE_Z();
    static std::string SIZE_T();
  };
}

#endif
