package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.request.RequestObject;
import org.dmg.pmml.DataDictionary;


public interface DataDictionaryInitializer {

    public DataDictionary init(RequestObject req);

}
