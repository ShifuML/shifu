package ml.shifu.plugin.mahout.adapter;

import java.util.List;

import ml.shifu.plugin.GenericMLModelBuilder;

import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionModel;

public class MahoutLogisticRegressionBuilder implements
        GenericMLModelBuilder<OnlineLogisticRegression, RegressionModel> {
    OnlineLogisticRegression mlModel;
    private RegressionModel pmmlModel;

    @Override
    public OnlineLogisticRegression createMLModelFromPMML(
            RegressionModel pmmlModel) {
        this.pmmlModel = pmmlModel;
        initNNLayer();
        setWeight();
        return mlModel;
    }

    private void initNNLayer() {
        mlModel = new OnlineLogisticRegression(2, pmmlModel
                .getRegressionTables().get(0).getNumericPredictors().size(),
                new L1());
    }

    private void setWeight() {
        List<NumericPredictor> nPredictors = pmmlModel.getRegressionTables()
                .get(0).getNumericPredictors();
        for (int i = 0; i < nPredictors.size(); i++) {
            mlModel.setBeta(0, i, nPredictors.get(i).getCoefficient());
        }
    }

}
