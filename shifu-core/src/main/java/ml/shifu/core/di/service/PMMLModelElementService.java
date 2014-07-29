package ml.shifu.core.di.service;


import com.google.inject.Inject;
import ml.shifu.core.di.spi.PMMLModelElementCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Model;

public class PMMLModelElementService {

    private PMMLModelElementCreator PMMLModelElementCreator;


    @Inject
    public PMMLModelElementService(PMMLModelElementCreator PMMLModelElementCreator) {
        this.PMMLModelElementCreator = PMMLModelElementCreator;

    }


    public Model getModelElement(Params params) {
        return PMMLModelElementCreator.create(params);
    }


}
