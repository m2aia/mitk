/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkControlPointManager_h
#define mitkControlPointManager_h

#include <MitkSemanticRelationsExports.h>

// semantic relations module
#include "mitkSemanticTypes.h"

// mitk core
#include <mitkDataNode.h>

namespace mitk
{
  /**
  * @brief Provides helper functions that are needed to work with control points.
  *
  *   These functions help to generate new control points, check for overlapping / containing control points or provide functionality
  *   to find a fitting control point or even extend an already existing control point.
  */
  /**
  * @brief Generates a control point from a given data node.
  *        The date is extracted from the data node by using the 'DICOMHelper::GetDICOMDateFromDataNode'-function.
  *
  * @param datanode   A data node pointer, whose date should be included in the newly generated control point.
  */
  MITKSEMANTICRELATIONS_EXPORT SemanticTypes::ControlPoint GenerateControlPoint(const mitk::DataNode* datanode);
  /**
  * @brief Find and return a whole control point including its date given a specific control point UID.
  *
  * @param caseID              The current case identifier is defined by the given string.
  * @param controlPointUID     The control point UID as string.
  *
  * @return   The control point with its UID and the date.
  */
  MITKSEMANTICRELATIONS_EXPORT SemanticTypes::ControlPoint GetControlPointByUID(const SemanticTypes::CaseID& caseID, const SemanticTypes::ID& controlPointUID);
  /**
  * @brief Returns an already existing control point from the given vector of control points. This existing control point has the
  *        the same date (year, month, day) as the given single control point.
  *        If no existing control point can be found an empty control point is returned.
  *
  * @param caseID           The current case identifier is defined by the given string.
  * @param controlPoint     The control point to check for existence.
  *
  * @return The existing control point.
  */
  MITKSEMANTICRELATIONS_EXPORT SemanticTypes::ControlPoint FindExistingControlPoint(const SemanticTypes::CaseID& caseID, const SemanticTypes::ControlPoint& controlPoint);
  /**
  * @brief Returns an already existing close control point from the given vector of control points. This closest control point has a
  *        date that is within a certain distance-in-days to the given control point.
  *        If no closest control point can be found within the distance threshold an empty control point is returned.
  *
  * @param caseID           The current case identifier is defined by the given string.
  * @param controlPoint     The control point to check for distance.
  *
  * @return The closest control point.
  */
  MITKSEMANTICRELATIONS_EXPORT SemanticTypes::ControlPoint FindClosestControlPoint(const SemanticTypes::CaseID& caseID, const SemanticTypes::ControlPoint& controlPoint);
  /**
  * @brief Returns the examination period to which the given control point belongs.
  *        Each examination point holds a vector of control point UIDs so that the UID of the given control point can be compared against the UIDs of the vector.
  *        An empty examination period is returned if,
  *        - the given vector of examination periods is empty
  *        - the examination periods do not contain any control point UIDs
  *        - the UID of the given control point is not contained in any examination period
  *
  * @param caseID           The current case identifier is defined by the given string.
  * @param controlPoint     The control point of which the examination period should be found.
  *
  * @return The examination period that contains the given control point.
  */
  MITKSEMANTICRELATIONS_EXPORT SemanticTypes::ExaminationPeriod FindContainingExaminationPeriod(const SemanticTypes::CaseID& caseID, const SemanticTypes::ControlPoint& controlPoint);
  /**
  * @brief Return the examination period to which the given data node belongs.
  *        The control point is used to find an already existing or the closest control point in the semantic relations storage.
  *        If such a control point is found, the 'FindClosestControlPoint'-function with this control point as an argument is used
  *        to actually find the corresponding examination period.
  *
  * @param caseID           The current case identifier is defined by the given string.
  * @param controlPoint     The control point of which the examination period should be found.
  *
  * @return The examination period that fits the given data node.
  */
  MITKSEMANTICRELATIONS_EXPORT SemanticTypes::ExaminationPeriod FindFittingExaminationPeriod(const SemanticTypes::CaseID& caseID, const SemanticTypes::ControlPoint& controlPoint);
  /**
  * @brief Return the examination period to which the given data node belongs.
  *        The DICOM date of the data node is used to find an already existing or the closest control point in the semantic relations storage.
  *        If such a control point is found, the 'FindFittingExaminationPeriod'-function with this control point as an argument is used
  *        to actually find the corresponding examination period.
  *
  * @param dataNode   A data node pointer, whose date should be included in the newly generated control point.
  *
  * @return The examination period that contains the given data node.
  */
  MITKSEMANTICRELATIONS_EXPORT SemanticTypes::ExaminationPeriod FindFittingExaminationPeriod(const DataNode* dataNode);
  /**
  * @brief Sort the given vector of examination periods.
  *        Each examination period has a vector of control point UIDs (stored in chronological order).
  *        The examination periods can be sorted by comparing the first control points of the examination periods.
  *
  * @param caseID                 The current case identifier is defined by the given string.
  * @param allExaminationPeriods  The examination periods to sort.
  */
  MITKSEMANTICRELATIONS_EXPORT void SortAllExaminationPeriods(const SemanticTypes::CaseID& caseID, SemanticTypes::ExaminationPeriodVector& allExaminationPeriods);

} // namespace mitk

#endif
