package ml.shifu.core.di.service;


import com.google.inject.Inject;
import ml.shifu.core.di.spi.PMMLLocalTransformationsCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.PMML;

public class PMMLLocalTransformationsService {
    @Inject
    private PMMLLocalTransformationsCreator localTransformationsCreator;

    public LocalTransformations createLocalTransformations(PMML pmml, Params params) throws Exception {
        return localTransformationsCreator.create(pmml, params);
    }


}
