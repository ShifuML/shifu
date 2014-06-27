package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.Targets;

public interface TargetsElementCreator {

    public Targets create(Params params);
}
