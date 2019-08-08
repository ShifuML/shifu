package ml.shifu.shifu.core;

import ml.shifu.shifu.core.dtrain.multitask.IndependentMTNNModel;
import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

import java.io.IOException;
import java.io.InputStream;

/**
 * @author haillu
 */
public class MTNNModel extends BasicML implements MLRegression {
    private IndependentMTNNModel independentMTNNModel;

    public MTNNModel(IndependentMTNNModel independentMTNNModel){
        this.independentMTNNModel = independentMTNNModel;
    }

    @Override
    public void updateProperties() {
        // No need to implement
    }

    @Override
    public MLData compute(MLData input) {
        double[] result = independentMTNNModel.compute(input.getData());
        return new BasicMLData(result);
    }

    @Override
    public int getInputCount() {
        return independentMTNNModel.getMtnn().getInputSize();
    }

    @Override
    public int getOutputCount() {
        return independentMTNNModel.getMtnn().getTaskNumber();
    }

    public static MTNNModel loadFromStream(InputStream input) throws IOException {
        return new MTNNModel(IndependentMTNNModel.loadFromStream(input));
    }

    public static MTNNModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        return new MTNNModel(IndependentMTNNModel.loadFromStream(input, isRemoveNameSpace));
    }


}
