package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.PMML;

public interface PMMLCreator {

    public PMML create(Params params);
}
