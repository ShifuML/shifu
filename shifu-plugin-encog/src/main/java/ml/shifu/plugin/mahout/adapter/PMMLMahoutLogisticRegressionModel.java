package ml.shifu.plugin.mahout.adapter;

import ml.shifu.plugin.PMMLAdapterCommonUtil;
import ml.shifu.plugin.PMMLModelBuilder;

import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.Matrix;
import org.dmg.pmml.RegressionModel;

public class PMMLMahoutLogisticRegressionModel implements
        PMMLModelBuilder<RegressionModel, OnlineLogisticRegression> {

    public RegressionModel adaptMLModelToPMML(OnlineLogisticRegression lrModel,
            RegressionModel pmmlModel) {
        Matrix matrix = lrModel.getBeta();
        int[] count = matrix.getNumNondefaultElements();
        double[] weights = new double[count[0]];
        for (int i = 0; i < count[0]; i++)
            weights[i] = matrix.get(0, i);

        return PMMLAdapterCommonUtil.getRegressionTable(weights, 0, pmmlModel);
    }

}
