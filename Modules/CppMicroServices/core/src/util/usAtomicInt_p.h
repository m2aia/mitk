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


#ifndef USATOMICINT_H
#define USATOMICINT_H

#include <usCoreConfig.h>
#include "usThreads_p.h"

US_BEGIN_NAMESPACE

/**
 * This class acts as an atomic integer.
 *
 * The integer value represented by this class can be incremented
 * and decremented atomically. This is often useful in reference
 * counting scenarios to minimize locking overhead in multi-threaded
 * environments.
 */
class AtomicInt : private AtomicCounter
{

public:

  AtomicInt(int value = 0) : AtomicCounter(value) {}

  /**
   * Increase the reference count atomically by 1.
   *
   * \return <code>true</code> if the new value is unequal to zero, <code>false</code>
   *         otherwise.
   */
  inline bool Ref() const
  { return AtomicIncrement() != 0; }

  /**
   * Decrease the reference count atomically by 1.
   *
   * \return <code>true</code> if the new value is unequal to zero, <code>false</code>
   *         otherwise.
   */
  inline bool Deref() const
  { return AtomicDecrement() != 0; }

  /**
   * Returns the current value.
   *
   */
  inline operator int() const
  {
    IntType curr(0);
    AtomicAssign(curr);
    return curr;
  }

};

US_END_NAMESPACE

#endif // USATOMICINT_H
