package ml.shifu.core.di.service;


import com.google.inject.Inject;
import ml.shifu.core.di.spi.MiningSchemaCreator;
import ml.shifu.core.request.RequestObject;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

public class MiningSchemaService {
    @Inject
    private MiningSchemaCreator miningSchemaCreator;


    public MiningSchema createMiningSchema(Model model, PMML pmml, RequestObject req) {
        return miningSchemaCreator.create(model, pmml, req);
    }


}
