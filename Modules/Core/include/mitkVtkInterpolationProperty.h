/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkVtkInterpolationProperty_h
#define mitkVtkInterpolationProperty_h

#include "mitkEnumerationProperty.h"

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  /**
   * Encapsulates the enumeration vtkInterpolation. Valid values are
   * (VTK constant/Id/string representation):
   * VTK_FLAT/0/Flat, VTK_GOURAUD/1/Gouraud, VTK_PHONG/2/Phong
   * Default is the Gouraud interpolation
   */
  class MITKCORE_EXPORT VtkInterpolationProperty : public EnumerationProperty
  {
  public:
    mitkClassMacro(VtkInterpolationProperty, EnumerationProperty);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

    mitkNewMacro1Param(VtkInterpolationProperty, const IdType &);

    mitkNewMacro1Param(VtkInterpolationProperty, const std::string &);

    /**
     * Returns the current interpolation value as defined by VTK constants.
     * @returns the current interpolation as VTK constant.
     */
    virtual int GetVtkInterpolation();

    /**
     * Sets the interpolation type to VTK_FLAT.
     */
    virtual void SetInterpolationToFlat();

    /**
     * Sets the interpolation type to VTK_WIREFRAME.
     */
    virtual void SetInterpolationToGouraud();

    /**
     * Sets the interpolation type to VTK_SURFACE.
     */
    virtual void SetInterpolationToPhong();

    using BaseProperty::operator=;

  protected:
    /**
     * Constructor. Sets the representation to a default value of surface(2)
     */
    VtkInterpolationProperty();

    /**
     * Constructor. Sets the interpolation to the given value. If it is not
     * valid, the interpolation is set to gouraud(1)
     * @param value the integer representation of the interpolation
     */
    VtkInterpolationProperty(const IdType &value);

    /**
     * Constructor. Sets the interpolation to the given value. If it is not
     * valid, the representation is set to gouraud(1)
     * @param value the string representation of the interpolation
     */
    VtkInterpolationProperty(const std::string &value);

    /**
     * this function is overridden as protected, so that the user may not add
     * additional invalid interpolation types.
     */
    bool AddEnum(const std::string &name, const IdType &id) override;

    /**
     * Adds the enumeration types as defined by vtk to the list of known
     * enumeration values.
     */
    virtual void AddInterpolationTypes();

  private:
    // purposely not implemented
    VtkInterpolationProperty &operator=(const VtkInterpolationProperty &);

    itk::LightObject::Pointer InternalClone() const override;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // end of namespace mitk

#endif
