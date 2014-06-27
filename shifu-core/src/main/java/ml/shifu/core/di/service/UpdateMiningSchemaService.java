package ml.shifu.core.di.service;


import com.google.inject.Inject;
import ml.shifu.core.di.spi.MiningSchemaUpdater;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Model;

public class UpdateMiningSchemaService {

    @Inject
    private MiningSchemaUpdater miningSchemaUpdater;


    public void updateMiningSchema(Model model, Params params) {
        miningSchemaUpdater.update(model, params);
    }

}
