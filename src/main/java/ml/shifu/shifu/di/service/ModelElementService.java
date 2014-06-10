package ml.shifu.shifu.di.service;


import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.TargetsElementCreator;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.Model;
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
