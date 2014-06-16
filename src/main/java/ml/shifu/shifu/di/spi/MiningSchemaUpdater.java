package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.util.Params;
import org.dmg.pmml.Model;

public interface MiningSchemaUpdater {

    public void update(Model model, Params params);

}
