package ml.shifu.shifu.di.service;


import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.MiningSchemaCreator;
import ml.shifu.shifu.di.spi.MiningSchemaUpdater;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

public class UpdateMiningSchemaService {

    @Inject
    private MiningSchemaUpdater miningSchemaUpdater;


    public void updateMiningSchema(Model model, Params params) {
        miningSchemaUpdater.update(model, params);
    }

}
