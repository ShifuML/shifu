package ml.shifu.shifu.core;

import java.util.List;

/**
 * Created by yliu15 on 2014/11/20.
 */
class DoubleVarsBinning implements AbstractBinning<Double> {

    private Estimator<Double> estimator;

    public DoubleVarsBinning(int maxBin){

        this.estimator = new Estimator<Double>(maxBin);
    }

    @Override
    public void clearBins() {
        this.estimator.clear();
    }

    @Override
    public void add(Double e) {
        this.estimator.add(e);
    }

    @Override
    public List<Double> getBins() {
        return this.estimator.getBin();
    }
}
