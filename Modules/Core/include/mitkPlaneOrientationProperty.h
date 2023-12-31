/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkPlaneOrientationProperty_h
#define mitkPlaneOrientationProperty_h

#include "mitkEnumerationProperty.h"

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  /**
   * Property which controls whether 2D line representation of a PlaneGeometry
   * should have small arrows at both ends to indicate the orientation of
   * the plane, and whether the arrows should be oriented in the direction of
   * the plane's normal or against it.
   *
   * Valid values of the enumeration property are
   * - PLANE_DECORATION_NONE (no arrows)
   * - PLANE_DECORATION_POSITIVE_ORIENTATION (arrows pointing upwards)
   * - PLANE_DECORATION_NEGATIVE_ORIENTATION (arrows pointing downwards)
   *
   * See also mitk::PlaneGeometryDataMapper2D::DrawOrientationArrow()
   */
  class MITKCORE_EXPORT PlaneOrientationProperty : public EnumerationProperty
  {
  public:
    mitkClassMacro(PlaneOrientationProperty, EnumerationProperty);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

    mitkNewMacro1Param(PlaneOrientationProperty, const IdType &);

    mitkNewMacro1Param(PlaneOrientationProperty, const std::string &);

    enum
    {
      PLANE_DECORATION_NONE,
      PLANE_DECORATION_POSITIVE_ORIENTATION,
      PLANE_DECORATION_NEGATIVE_ORIENTATION
    };

    /**
     * Returns the state of plane decoration.
     */
    virtual int GetPlaneDecoration();

    /**
     * Sets the decoration type to no decoration.
     */
    virtual void SetPlaneDecorationToNone();

    /**
     * Sets the decoration type to arrows in positive plane direction.
     */
    virtual void SetPlaneDecorationToPositiveOrientation();

    /**
     * Sets the decoration type to arrows in negative plane direction.
     */
    virtual void SetPlaneDecorationToNegativeOrientation();

    using BaseProperty::operator=;

  protected:
    /**
     * Constructor. Sets the decoration type to none.
     */
    PlaneOrientationProperty();

    /**
     * Constructor. Sets the decoration type to the given value. If it is not
     * valid, the interpolation is set to none
     */
    PlaneOrientationProperty(const IdType &value);

    /**
     * Constructor. Sets the decoration type to the given value. If it is not
     * valid, the representation is set to none
     */
    PlaneOrientationProperty(const std::string &value);

    /**
     * this function is overridden as protected, so that the user may not add
     * additional invalid types.
     */
    bool AddEnum(const std::string &name, const IdType &id) override;

    /**
     * Adds the standard enumeration types with corresponding strings.
     */
    virtual void AddDecorationTypes();

  private:
    // purposely not implemented
    PlaneOrientationProperty &operator=(const PlaneOrientationProperty &);

    itk::LightObject::Pointer InternalClone() const override;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // end of namespace mitk

#endif
