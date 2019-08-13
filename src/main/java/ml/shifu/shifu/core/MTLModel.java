package ml.shifu.shifu.core;

import ml.shifu.shifu.core.dtrain.mtl.IndependentMTLModel;
import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

import java.io.IOException;
import java.io.InputStream;

/**
 * @author haillu
 */
public class MTLModel extends BasicML implements MLRegression {
    private IndependentMTLModel independentMTLModel;

    public MTLModel(IndependentMTLModel independentMTLModel){
        this.independentMTLModel = independentMTLModel;
    }

    @Override
    public void updateProperties() {
        // No need to implement
    }

    @Override
    public MLData compute(MLData input) {
        double[] result = independentMTLModel.compute(input.getData());
        return new BasicMLData(result);
    }

    @Override
    public int getInputCount() {
        return independentMTLModel.getMtl().getInputSize();
    }

    @Override
    public int getOutputCount() {
        return independentMTLModel.getMtl().getTaskNumber();
    }

    public static MTLModel loadFromStream(InputStream input) throws IOException {
        return new MTLModel(IndependentMTLModel.loadFromStream(input));
    }

    public static MTLModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        return new MTLModel(IndependentMTLModel.loadFromStream(input, isRemoveNameSpace));
    }


}
