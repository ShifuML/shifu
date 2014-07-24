package ml.shifu.core.di.service;

import com.google.inject.Inject;
import ml.shifu.core.di.spi.PMMLDataDictionaryCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.DataDictionary;

public class PMMLDataDictionaryService {

    private PMMLDataDictionaryCreator PMMLDataDictionaryCreator;


    @Inject
    public PMMLDataDictionaryService(PMMLDataDictionaryCreator PMMLDataDictionaryCreator) {
        this.PMMLDataDictionaryCreator = PMMLDataDictionaryCreator;
    }

    public DataDictionary getDataDictionary(Params params) {
        return PMMLDataDictionaryCreator.create(params);
    }
}
