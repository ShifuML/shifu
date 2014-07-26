package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.DataDictionary;


public interface PMMLDataDictionaryCreator {

    public DataDictionary create(Params params);

}
