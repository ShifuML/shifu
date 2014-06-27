package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.Model;

public interface MiningSchemaUpdater {

    public void update(Model model, Params params);

}
