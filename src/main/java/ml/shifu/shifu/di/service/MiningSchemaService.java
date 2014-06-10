package ml.shifu.shifu.di.service;


import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.MiningSchemaCreator;
import ml.shifu.shifu.di.spi.TargetsElementCreator;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Targets;

public class MiningSchemaService {

    private MiningSchemaCreator miningSchemaCreator;

    @Inject
    public MiningSchemaService(MiningSchemaCreator miningSchemaCreator) {
        this.miningSchemaCreator = miningSchemaCreator;
    }

    public MiningSchema createMiningSchema(Model model, PMML pmml, RequestObject req) {
        return miningSchemaCreator.create(model, pmml, req);
    }

}
