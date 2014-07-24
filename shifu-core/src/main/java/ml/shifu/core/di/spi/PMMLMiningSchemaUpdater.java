package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.Model;

public interface PMMLMiningSchemaUpdater {

    public void update(Model model, Params params);

}
