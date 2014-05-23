package ml.shifu.shifu.di.spi;

import java.util.Map;
import ml.shifu.shifu.pmml.obj.DataDictionary;

public interface DataDictionaryTypeSetter {

    public void setType(DataDictionary dict, Map<String, Object> params);

}
