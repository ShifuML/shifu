package ml.shifu.core.di.spi;

import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.Params;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

public interface PMMLMiningSchemaCreator {

    public MiningSchema create(PMML pmml, Params params) throws Exception;
}
