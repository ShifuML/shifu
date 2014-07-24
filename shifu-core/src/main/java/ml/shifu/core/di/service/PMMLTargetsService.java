package ml.shifu.core.di.service;

import com.google.inject.Inject;
import ml.shifu.core.di.spi.PMMLTargetsCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Targets;

public class PMMLTargetsService {

    private PMMLTargetsCreator PMMLTargetsCreator;

    @Inject
    public PMMLTargetsService(PMMLTargetsCreator PMMLTargetsCreator) {
        this.PMMLTargetsCreator = PMMLTargetsCreator;
    }


    public Targets createTargetsElement(PMML pmml, Params params) {
        return PMMLTargetsCreator.create(pmml, params);
    }

}
