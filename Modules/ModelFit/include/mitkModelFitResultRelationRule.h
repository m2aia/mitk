/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkModelFitResultRelationRule_h
#define mitkModelFitResultRelationRule_h

#include "mitkSourceImageRelationRule.h"

#include "MitkModelFitExports.h"

namespace mitk
{
  /**This rule class can be used to represent the relation between an image generated by a model fit (rule source)
  and the image that was fitted (rule destination).
  The addition to the behavirID-layer is supported like for SourceImageRelationRule.
  In addition the rule uses the data-layer to deduce/define relations.
  For this layer it uses properties compliant to DICOM. Thus (1) the information is stored in
  a DICOM Source Image Sequence item (0x0008,0x2112) and (2) the destination must have properties
  DICOM SOP Instance UIDs (0x0008, 0x0018) and DICOM SOP Class UID (0x0008, 0x0016). If the destination
  does not have this properties, no connection can be made on the data-layer.
  @remark Please note that PropertyRelationRules and DICOM use the term "source" differently. The DICOM
  source (image) equals the PropertyRelationRule destination. This is due to
  an inverted relation direction. So in the context of the SourceImageRelationRule interface a derived data is
  the source and points to the original image, it derives from. In the context of DICOM this referenced original image would be
  called source image (as the name of this class).
  In order to be able to use this class for different relation types (DICOM would call it purposes),
  the purposeTag is used. It must be specified when creating a rule instance.
  The purposeTag will be used as suffix for the rule ID of the instance and therefore allows to create
  specific and distinguishable rules instances based on this class.
  One may also specify the display name and the role names of the instance. If not specified the default values
  are used (display name: "<purposeTag> relation", source role name: "derived data",
  destination role name: "source image")
  */
  class MITKMODELFIT_EXPORT ModelFitResultRelationRule : public mitk::SourceImageRelationRule
  {
  public:
    mitkClassMacro(ModelFitResultRelationRule, SourceImageRelationRule);
    itkNewMacro(Self);

    using RuleIDType = Superclass::RuleIDType;
    using RelationUIDType = Superclass::RelationUIDType;
    using RelationUIDVectorType = Superclass::RelationUIDVectorType;

    bool IsDestinationCandidate(const IPropertyProvider *owner) const override;

  protected:
    ModelFitResultRelationRule();
    ~ModelFitResultRelationRule() override = default;

    using InstanceIDType = SourceImageRelationRule::InstanceIDType;
    using InstanceIDVectorType = SourceImageRelationRule::InstanceIDVectorType;

    using DataRelationUIDVectorType = Superclass::DataRelationUIDVectorType;
    virtual DataRelationUIDVectorType GetRelationUIDs_DataLayer(const IPropertyProvider* source,
      const IPropertyProvider* destination, const InstanceIDVectorType& instances_IDLayer) const override;

    void Disconnect_datalayer(IPropertyOwner *source, const RelationUIDType& relationUID) const override;

    itk::LightObject::Pointer InternalClone() const override;
  };

} // namespace mitk

#endif
