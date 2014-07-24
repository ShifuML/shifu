package ml.shifu.core.request.processor.deprecated;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.PMMLDataDictionaryService;
import ml.shifu.core.request.Binding;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.PMML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CreateDataDictionaryRequestProcessor {

    private static Logger log = LoggerFactory.getLogger(CreateDataDictionaryRequestProcessor.class);

    public static void run(Request req) {
    /*
        log.info("Starting Processor... ");


        Params requestParams = req.getParams();

        String pathPMML = requestParams.get("pathPMML", "./model.xml").toString();

        Binding binding = null;

        for (Binding b : req.getBindings()) {
            if (b.getSpi().equals("DataDictionaryInitializer")) {
                if (binding == null) {
                    binding = b;
                } else {
                    throw new RuntimeException("Binding should be unique: DataDictionaryInitializer");
                }
            }
        }

        if (binding == null) {
            throw new RuntimeException("Missing binding for: DataDictionaryInitializer");
        }


        SimpleModule module = new SimpleModule();
        module.set(binding.getSpi(), binding.getImpl());

        //module.setBindings((Map<String, String>) req.getGlobalParams().get("bindings"));
        Injector injector = Guice.createInjector(module);

        Params params = binding.getParams();

        PMMLDataDictionaryService dataDictionaryService = injector.getInstance(PMMLDataDictionaryService.class);
        DataDictionary dataDictionary = dataDictionaryService.getDataDictionary(params);


        PMML pmml = new PMML();
        pmml.setDataDictionary(dataDictionary);

        PMMLUtils.savePMML(pmml, pathPMML);                   */
    }
}
