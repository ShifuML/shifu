package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;

public interface MiningSchemaCreator {

    public MiningSchema create(Model model, PMML pmml, RequestObject req);
}
