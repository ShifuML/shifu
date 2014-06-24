package ml.shifu.shifu.di.service;


import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.MiningSchemaUpdater;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.Model;

public class UpdateMiningSchemaService {

    @Inject
    private MiningSchemaUpdater miningSchemaUpdater;


    public void updateMiningSchema(Model model, Params params) {
        miningSchemaUpdater.update(model, params);
    }

}
