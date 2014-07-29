package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.PMML;

public interface PMMLLocalTransformationsCreator {

    public LocalTransformations create(PMML pmml, Params params) throws Exception;
}
