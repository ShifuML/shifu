package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Targets;

public interface PMMLTargetsCreator {

    public Targets create(PMML pmml, Params params);
}
