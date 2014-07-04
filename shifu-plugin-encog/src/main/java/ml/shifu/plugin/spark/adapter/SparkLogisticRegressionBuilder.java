package ml.shifu.plugin.spark.adapter;

import java.util.List;

import ml.shifu.plugin.GenericMLModelBuilder;

import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionTable;

public class SparkLogisticRegressionBuilder implements
        GenericMLModelBuilder<LogisticRegressionModel, RegressionModel> {
    LogisticRegressionModel mlModel;

    @Override
    public LogisticRegressionModel createMLModelFromPMML(
            RegressionModel pmmlModel) {

        RegressionTable rTable = pmmlModel.getRegressionTables()
                .get(0);
        List<NumericPredictor> nPredictors = rTable.getNumericPredictors();
        double intercept = rTable.getIntercept();
        double[] coefficients = new double[nPredictors.size()+1];
        coefficients[0]=intercept;
        for (int i = 0; i < nPredictors.size(); i++) {
           coefficients[i+1] = nPredictors.get(i).getCoefficient();
        }
        Vector vector = new DenseVector(coefficients);
        mlModel =new   LogisticRegressionModel(vector,intercept);
        mlModel.clearThreshold();
        return mlModel;
    }

}
