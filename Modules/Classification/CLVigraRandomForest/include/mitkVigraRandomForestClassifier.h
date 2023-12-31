/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkVigraRandomForestClassifier_h
#define mitkVigraRandomForestClassifier_h

#include <MitkCLVigraRandomForestExports.h>
#include <mitkAbstractClassifier.h>

//#include <vigra/multi_array.hxx>
#include <vigra/random_forest.hxx>

#include <mitkBaseData.h>

namespace mitk
{
  class MITKCLVIGRARANDOMFOREST_EXPORT VigraRandomForestClassifier : public AbstractClassifier
  {
  public:

    mitkClassMacro(VigraRandomForestClassifier, AbstractClassifier);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

      VigraRandomForestClassifier();

    ~VigraRandomForestClassifier() override;

    void Train(const Eigen::MatrixXd &X, const Eigen::MatrixXi &Y) override;
    void OnlineTrain(const Eigen::MatrixXd &X, const Eigen::MatrixXi &Y);
    Eigen::MatrixXi Predict(const Eigen::MatrixXd &X) override;
    Eigen::MatrixXi PredictWeighted(const Eigen::MatrixXd &X);


    bool SupportsPointWiseWeight() override;
    bool SupportsPointWiseProbability() override;
    void ConvertParameter();

    void SetRandomForest(const vigra::RandomForest<int> & rf);
    const vigra::RandomForest<int> & GetRandomForest() const;

    void UsePointWiseWeight(bool) override;
    void SetMaximumTreeDepth(int);
    void SetMinimumSplitNodeSize(int);
    void SetPrecision(double);
    void SetSamplesPerTree(double);
    void UseSampleWithReplacement(bool);
    void SetTreeCount(int);
    void SetWeightLambda(double);

    void SetTreeWeights(Eigen::MatrixXd weights);
    void SetTreeWeight(int treeId, double weight);
    Eigen::MatrixXd GetTreeWeights() const;

    void PrintParameter(std::ostream &str = std::cout);

  private:
    // *-------------------
    // * THREADING
    // *-------------------


    struct TrainingData;
    struct PredictionData;
    struct EigenToVigraTransform;
    struct Parameter;

    vigra::MultiArrayView<2, double> m_Probabilities;
    Eigen::MatrixXd m_TreeWeights;

    Parameter * m_Parameter;
    vigra::RandomForest<int> m_RandomForest;

    static itk::ITK_THREAD_RETURN_TYPE TrainTreesCallback(void *);
    static itk::ITK_THREAD_RETURN_TYPE PredictCallback(void *);
    static itk::ITK_THREAD_RETURN_TYPE PredictWeightedCallback(void *);
    static void VigraPredictWeighted(PredictionData *data, vigra::MultiArrayView<2, double> & X, vigra::MultiArrayView<2, int> & Y, vigra::MultiArrayView<2, double> & P);
  };
}

#endif
