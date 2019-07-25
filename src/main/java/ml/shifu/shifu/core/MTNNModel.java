package ml.shifu.shifu.core;

import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;

/**
 * @author haillu
 */
public class MTNNModel extends BasicML implements MLRegression {
    @Override
    public void updateProperties() {

    }

    @Override
    public MLData compute(MLData input) {
        return null;
    }

    @Override
    public int getInputCount() {
        return 0;
    }

    @Override
    public int getOutputCount() {
        return 0;
    }
}
