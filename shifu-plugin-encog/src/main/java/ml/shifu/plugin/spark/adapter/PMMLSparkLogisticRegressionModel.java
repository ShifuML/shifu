package ml.shifu.plugin.spark.adapter;

import ml.shifu.plugin.PMMLAdapterCommonUtil;
import ml.shifu.plugin.PMMLModelBuilder;

/**
 * The class that converts the Spark LogisticRegressionModel to a PMML
 * RegressionModel. This class extends the abstract class
 * PMMLModelBuilder<pmml.RegressionModel,spark.LogisticRegressionModel>.
 * 
 */
public class PMMLSparkLogisticRegressionModel implements
        PMMLModelBuilder<org.dmg.pmml.RegressionModel, org.apache.spark.mllib.classification.LogisticRegressionModel> {
   
    /**
     * The function which converts the Spark LogisticRegressionModel to a PMML
     * RegressionModel.
     * 
     * @param lrModel
     *            Spark LogisticRegressionModel
     * @param utility
     *            DataFieldUtility that provides supplementary data field for
     *            the model conversion
     * 
     * @return The generated PMML RegressionModel
     */
    public org.dmg.pmml.RegressionModel adaptMLModelToPMML(org.apache.spark.mllib.classification.LogisticRegressionModel lrModel,
            org.dmg.pmml.RegressionModel pmmlModel) {
        double[] weights = lrModel.weights().toArray();
        double intercept = weights[0];// lrModel.intercept();
        
        return PMMLAdapterCommonUtil.getRegressionTable(weights, intercept, pmmlModel);
    }

}
