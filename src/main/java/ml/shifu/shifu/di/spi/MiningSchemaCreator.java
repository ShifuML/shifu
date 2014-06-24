package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.request.RequestObject;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

public interface MiningSchemaCreator {

    public MiningSchema create(Model model, PMML pmml, RequestObject req);
}
