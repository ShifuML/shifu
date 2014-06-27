package ml.shifu.core.di.spi;

import ml.shifu.core.request.RequestObject;
import org.dmg.pmml.DataDictionary;


public interface DataDictionaryInitializer {

    public DataDictionary init(RequestObject req);

}
