/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkLesionData_h
#define mitkLesionData_h

#include <MitkSemanticRelationsExports.h>

// mitk semantic relations module
#include "mitkSemanticTypes.h"

// c++
#include <vector>

namespace mitk
{
  /**
  * @brief This class holds the data of each lesion in the lesion tree view.
  *        The data is the lesion itself with its UID, name and lesion class
  *        as well as two vectors for
  *        - lesion presence: bool value for each control-point
  *                           inside the semantic relations storage
  *        - lesion volume: double value for each control-point  - information type pair
  *                         inside the semantic relations storage
  *
  */
  class MITKSEMANTICRELATIONS_EXPORT LesionData
  {
  public:
    /**
    * @brief sets the data members to their initial values
    */
    LesionData(const SemanticTypes::Lesion& lesion = SemanticTypes::Lesion());

    SemanticTypes::Lesion GetLesion() const { return m_Lesion; };
    SemanticTypes::ID GetLesionUID() const { return m_Lesion.UID; }
    std::string GetLesionName() const { return m_Lesion.name; }

    const std::vector<bool>& GetLesionPresence() const { return m_LesionPresence; };
    const std::vector<double>& GetLesionVolume() const { return m_LesionVolume; };

    void SetLesion(const SemanticTypes::Lesion& lesion);
    void SetLesionPresence(const std::vector<bool>& lesionPresence);
    void SetLesionVolume(const std::vector<double>& lesionVolume);

  private:

    SemanticTypes::Lesion m_Lesion;
    std::vector<bool> m_LesionPresence;
    std::vector<double> m_LesionVolume;

  };
} // end namespace

#endif
