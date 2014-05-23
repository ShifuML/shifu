package ml.shifu.shifu.di.spi;

import java.util.Map;
import ml.shifu.shifu.pmml.obj.DataDictionary;

public interface DataDictionaryInitializer {

    public DataDictionary init(Map<String, Object> params);

}
