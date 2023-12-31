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

#ifndef USLDAPPROP_H
#define USLDAPPROP_H

#include <usCoreConfig.h>
#include <usAny.h>

US_BEGIN_NAMESPACE

/// \cond
class US_Core_EXPORT LDAPPropExpr
{
public:

  explicit LDAPPropExpr(const std::string& expr);

  LDAPPropExpr& operator!();

  operator std::string() const;

  bool IsNull() const;

private:

  LDAPPropExpr& operator=(const LDAPPropExpr&);

  std::string m_ldapExpr;
};
/// \endcond

/**
 * \ingroup MicroServicesUtils
 *
 * A fluent API for creating LDAP filter strings.
 *
 * Examples for creating LDAPFilter objects:
 * \code
 * // This creates the filter "(&(name=Ben)(!(count=1)))"
 * LDAPFilter filter(LDAPProp("name") == "Ben" && !(LDAPProp("count") == 1));
 *
 * // This creates the filter "(|(presence=*)(!(absence=*)))"
 * LDAPFilter filter(LDAPProp("presence") || !LDAPProp("absence"));
 *
 * // This creates the filter "(&(ge>=-3)(approx~=hi))"
 * LDAPFilter filter(LDAPProp("ge") >= -3 && LDAPProp("approx").Approx("hi"));
 * \endcode
 *
 * \sa LDAPFilter
 */
class US_Core_EXPORT LDAPProp
{
public:

  /**
   * Create a LDAPProp instance for the named LDAP property.
   *
   * @param property The name of the LDAP property.
   */
  LDAPProp(const std::string& property);

  /**
   * LDAP equality '='
   *
   * @param s A type convertible to std::string.
   * @return A LDAP expression object.
   *
   * @{
   */
  LDAPPropExpr operator==(const std::string& s) const;
  LDAPPropExpr operator==(const us::Any& s) const;
  template<class T>
  LDAPPropExpr operator==(const T& s) const
  {
    std::stringstream ss;
    ss << s;
    return LDAPPropExpr("(" + m_property + "=" + ss.str() + ")");
  }
  /// @}

  operator LDAPPropExpr () const;

  /**
   * States the absence of the LDAP property.
   *
   * @return A LDAP expression object.
   */
  LDAPPropExpr operator!() const;

  /**
   * Convenience operator for LDAP inequality.
   *
   * Writing either
   * \code
   * LDAPProp("attr") != "val"
   * \endcode
   * or
   * \code
   * !(LDAPProp("attr") == "val")
   * \endcode
   * leads to the same string "(!(attr=val))".
   *
   * @param s A type convertible to std::string.
   * @return A LDAP expression object.
   *
   * @{
   */
  LDAPPropExpr operator!=(const std::string& s) const;
  LDAPPropExpr operator!=(const us::Any& s) const;
  template<class T>
  LDAPPropExpr operator!=(const T& s) const
  {
    std::stringstream ss;
    ss << s;
    return operator!=(ss.str());
  }
  /// @}

  /**
   * LDAP greater or equal '>='
   *
   * @param s A type convertible to std::string.
   * @return A LDAP expression object.
   *
   * @{
   */
  LDAPPropExpr operator>=(const std::string& s) const;
  LDAPPropExpr operator>=(const us::Any& s) const;
  template<class T>
  LDAPPropExpr operator>=(const T& s) const
  {
    std::stringstream ss;
    ss << s;
    return operator>=(ss.str());
  }
  /// @}

  /**
   * LDAP less or equal '<='
   *
   * @param s A type convertible to std::string.
   * @return A LDAP expression object.
   *
   * @{
   */
  LDAPPropExpr operator<=(const std::string& s) const;
  LDAPPropExpr operator<=(const us::Any& s) const;
  template<class T>
  LDAPPropExpr operator<=(const T& s) const
  {
    std::stringstream ss;
    ss << s;
    return operator<=(ss.str());
  }
  /// @}

  /**
   * LDAP approximation '~='
   *
   * @param s A type convertible to std::string.
   * @return A LDAP expression object.
   *
   * @{
   */
  LDAPPropExpr Approx(const std::string& s) const;
  LDAPPropExpr Approx(const us::Any& s) const;
  template<class T>
  LDAPPropExpr Approx(const T& s) const
  {
    std::stringstream ss;
    ss << s;
    return Approx(ss.str());
  }
  /// @}

private:

  LDAPProp& operator=(const LDAPProp&);

  std::string m_property;
};

US_END_NAMESPACE

/**
 * \ingroup MicroServicesUtils
 *
 * LDAP logical and '&'
 *
 * @param left A LDAP expression.
 * @param right A LDAP expression.
 * @return A LDAP expression
 */
US_Core_EXPORT us::LDAPPropExpr operator&&(const us::LDAPPropExpr& left, const us::LDAPPropExpr& right);

/**
 * \ingroup MicroServicesUtils
 *
 * LDAP logical or '|'
 *
 * @param left A LDAP expression.
 * @param right A LDAP expression.
 * @return A LDAP expression
 */
US_Core_EXPORT us::LDAPPropExpr operator||(const us::LDAPPropExpr& left, const us::LDAPPropExpr& right);

#endif // USLDAPPROP_H
