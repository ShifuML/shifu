package ml.shifu.core.di.service;


import com.google.inject.Inject;
import ml.shifu.core.di.spi.PMMLMiningSchemaCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.PMML;

public class PMMLMiningSchemaService {
    @Inject
    private PMMLMiningSchemaCreator PMMLMiningSchemaCreator;


    public MiningSchema createMiningSchema(PMML pmml, Params params) throws Exception {
        return PMMLMiningSchemaCreator.create(pmml, params);

    }


}
