package ml.shifu.core.di.service;


import com.google.inject.Inject;
import ml.shifu.core.di.spi.TargetsElementCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Targets;

public class ModelElementService {

    private TargetsElementCreator targetsElementCreator;

    @Inject
    public ModelElementService(TargetsElementCreator targetsElementCreator) {
        this.targetsElementCreator = targetsElementCreator;
    }

    public Targets getTargetsElement(Params params) {
        return targetsElementCreator.create(params);
    }


}
