package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.util.Params;
import org.dmg.pmml.Targets;

public interface TargetsElementCreator {

    public Targets create(Params params);
}
