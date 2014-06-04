package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.util.Params;
import org.dmg.pmml.DataDictionary;

import java.util.Map;


public interface DataDictionaryInitializer {

    public DataDictionary init(Params params);

}
