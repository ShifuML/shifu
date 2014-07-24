package ml.shifu.core.di.spi;

import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Model;

public interface Trainer {

    public void train(Model pmmlMode, PMMLDataSet pmmlDataSet, Params params) throws Exception;

}
