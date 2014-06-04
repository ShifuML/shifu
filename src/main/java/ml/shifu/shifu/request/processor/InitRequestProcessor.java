package ml.shifu.shifu.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.DataDictionaryService;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.PMMLUtils;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.PMML;

public class InitRequestProcessor {

    public void run(RequestObject req) {

        SimpleModule module = new SimpleModule();
        module.setBindings(req.getBindings());
        Injector injector = Guice.createInjector(module);

        DataDictionaryService dataDictionaryService= injector.getInstance(DataDictionaryService.class);
        DataDictionary dataDictionary = dataDictionaryService.getDataDictionary(req.getParams());


        PMML pmml = new PMML();
        pmml.setDataDictionary(dataDictionary);
        PMMLUtils.savePMML(pmml, (String)req.getParams().get("pathPMML", "model.xml"));

    }
}
