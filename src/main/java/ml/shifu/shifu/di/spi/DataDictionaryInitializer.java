package ml.shifu.shifu.di.spi;

import org.dmg.pmml.DataDictionary;

import java.util.Map;


public interface DataDictionaryInitializer {

    public DataDictionary init(Map<String, Object> params);

}
