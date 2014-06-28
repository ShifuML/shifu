package ml.shifu.plugin.encog.adapter;

import java.util.HashMap;
import java.util.List;

import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionNormalizationMethodType;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

import pmmlAdapter.GenericMLModelBuilder;

public class EncogLogisticRegressionBuilder implements
        GenericMLModelBuilder<BasicNetwork, RegressionModel> {
    BasicNetwork mlModel = new BasicNetwork();
    private RegressionModel pmmlModel;
   @SuppressWarnings("serial")
    HashMap<RegressionNormalizationMethodType, ActivationFunction> functionMap = new HashMap<RegressionNormalizationMethodType, ActivationFunction>() {
        {
            put(RegressionNormalizationMethodType.LOGIT,
                    new ActivationSigmoid());
            put(RegressionNormalizationMethodType.SOFTMAX,
                    new ActivationSigmoid());
            put(RegressionNormalizationMethodType.NONE, new ActivationLinear());
        }
    };

    @Override
    public BasicNetwork createMLModelFromPMML(RegressionModel pmmlModel) {
        this.pmmlModel = pmmlModel;
        initNNLayer();
        setWeight();

        return mlModel;
    }

    private ActivationFunction transformActivationFunction(
            RegressionNormalizationMethodType pmmlActivationFuncType) {
        return functionMap.get(pmmlActivationFuncType);
    }

    private void initNNLayer() {
        mlModel.addLayer(new BasicLayer(new ActivationLinear(), true, pmmlModel
                .getRegressionTables().get(0).getNumericPredictors().size()));
        mlModel.addLayer(new BasicLayer(transformActivationFunction(pmmlModel
                .getNormalizationMethod()), false,1));
        mlModel.getStructure().finalizeStructure();
    }

    private void setWeight() {
        List<NumericPredictor> nPredictors = pmmlModel.getRegressionTables()
                .get(0).getNumericPredictors();
        for (int i=0;i<nPredictors.size();i++) {
            mlModel.setWeight(0, i, 0, nPredictors.get(i).getCoefficient());
        }
    }

}
