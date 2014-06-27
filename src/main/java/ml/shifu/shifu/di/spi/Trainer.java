package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.container.PMMLDataSet;
import ml.shifu.shifu.util.Params;

public interface Trainer {

    public void train(PMMLDataSet pmmlDataSet, Params params) throws Exception;

}
