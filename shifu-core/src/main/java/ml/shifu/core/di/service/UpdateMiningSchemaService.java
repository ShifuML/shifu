package ml.shifu.core.di.service;


import com.google.inject.Inject;
import ml.shifu.core.di.spi.PMMLMiningSchemaUpdater;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Model;

public class UpdateMiningSchemaService {

    @Inject
    private PMMLMiningSchemaUpdater PMMLMiningSchemaUpdater;


    public void updateMiningSchema(Model model, Params params) {
        PMMLMiningSchemaUpdater.update(model, params);
    }

}
