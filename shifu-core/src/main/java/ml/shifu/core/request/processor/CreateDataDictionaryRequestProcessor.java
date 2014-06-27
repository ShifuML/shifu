package ml.shifu.core.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.DataDictionaryService;
import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.PMML;

import java.util.Map;

public class CreateDataDictionaryRequestProcessor {

    public static void run(RequestObject req) {

        SimpleModule module = new SimpleModule();
        module.setBindings((Map<String, String>) req.getGlobalParams().get("bindings"));
        Injector injector = Guice.createInjector(module);

        Params params = req.getGlobalParams();

        DataDictionaryService dataDictionaryService = injector.getInstance(DataDictionaryService.class);
        DataDictionary dataDictionary = dataDictionaryService.getDataDictionary(req);

        PMML pmml = new PMML();
        pmml.setDataDictionary(dataDictionary);

        PMMLUtils.savePMML(pmml, (String) params.get("pathPMML"));

    }
}
