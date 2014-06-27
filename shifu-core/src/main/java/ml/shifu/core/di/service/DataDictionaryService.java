package ml.shifu.core.di.service;

import com.google.inject.Inject;
import ml.shifu.core.di.spi.DataDictionaryInitializer;
import ml.shifu.core.request.RequestObject;
import org.dmg.pmml.DataDictionary;

public class DataDictionaryService {

    private DataDictionaryInitializer initializer;


    @Inject
    public DataDictionaryService(DataDictionaryInitializer initializer) {
        this.initializer = initializer;
    }

    public DataDictionary getDataDictionary(RequestObject req) {
        return initializer.init(req);
    }
}
