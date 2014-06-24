package ml.shifu.shifu.di.service;


import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.MiningSchemaCreator;
import ml.shifu.shifu.request.RequestObject;
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
